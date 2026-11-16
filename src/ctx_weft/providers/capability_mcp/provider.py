"""MCPCapabilityProvider：用官方 `mcp` SDK 桥接 MCP server 与 ctx-weft capability 接口。

参照 miniAgents `app/tools` 的做法，直接复用 `mcp.client`（ClientSession +
stdio_client / streamablehttp_client），不再自行实现 JSON-RPC / SSE。

连接策略（关键）：
  - 连接由一个独立的 owner task（`_session_runner`）持有——进入与退出都在同一个 task
    内完成，避免 anyio cancel scope「跨 task 退出」的错误。
  - **绝不阻塞 agent loop**：`list()` / `invoke()` 走 `_ensure_connected()`，它只在后台
    起一次连接尝试后立即返回；未连上时 `list()` 返回 []、`invoke()` 立刻报错降级，工具
    会在后续某轮连上后自动出现。
  - **熔断/退避**：连接失败记 `cooldown_until` + 指数退避（封顶），冷却期内不再尝试，
    避免每个 turn 反复阻塞重连。
  - **预热**：`start()` 在服务启动时（非 agent 路径）主动连一次，让首轮就能拿到工具。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

from ctx_weft.protocols.capability import (
    Capability,
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.context import ProviderContext
from ctx_weft.protocols.content import ContentPart, ImagePart, TextPart
import httpx

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp import types as mcp_types

logger = logging.getLogger(__name__)

# MCP clientInfo 默认值：core 自身的中性身份。上层 host 可经 __init__ 注入覆盖
# （例如注入产品名）。core 不应硬编码任何 host/产品品牌。
_DEFAULT_CLIENT_NAME = "ctx-weft"
_DEFAULT_CLIENT_VERSION = "0.1.0"
_SIDE_EFFECT_KEYWORDS = ("write", "delete", "exec", "create")


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server connection."""

    name: str
    transport: Literal["stdio", "http", "streamable_http"] = "stdio"
    command: list[str] = field(default_factory=list)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    default_purposes: list[str] = field(default_factory=lambda: ["act"])
    capability_purpose_override: dict[str, list[str]] = field(default_factory=dict)
    timeout_per_call_sec: int = 60
    connect_timeout_sec: int = 10
    # 仅 http/streamable_http 有意义：底层 httpx 客户端是否信任环境变量
    # （HTTP_PROXY/HTTPS_PROXY/NO_PROXY/SSL_CERT_FILE/NETRC…）。默认 False＝直连、忽略
    # 系统代理与证书设置（内网部署常态）。core 自身不消费此字段，由 host 的 transport 实现读取。
    trust_env: bool = False
    ssl_verify: bool | str = False  # True / False / CA bundle path


class MCPCapabilityProvider(ToolCapabilityProvider):
    """Bridges an MCP server to the ctx-weft capability interface via the `mcp` SDK.

    Must inherit `ToolCapabilityProvider` (not bare `CapabilityProvider`): the gateway
    builds its router index via `isinstance(p, ToolCapabilityProvider)`, so a bare base
    would silently drop every MCP tool from routing while the cache still advertises them
    (→ "no provider found for 'mcp:...'").
    """

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        max_reconnect_attempts: int = 3,
        reconnect_base_delay_sec: float = 1.0,
        client_name: str = _DEFAULT_CLIENT_NAME,
        client_version: str = _DEFAULT_CLIENT_VERSION,
    ) -> None:
        self._cfg = config
        self._client_name = client_name
        self._client_version = client_version
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_base_delay_sec = reconnect_base_delay_sec
        # 退避封顶：base * 2^attempts（默认 1×2^3 = 8s）。失败后冷却时长在此上限内翻倍。
        self._max_backoff_sec = max(
            reconnect_base_delay_sec, reconnect_base_delay_sec * (2 ** max_reconnect_attempts)
        )
        self.name = f"mcp:{config.name}"
        # 由 session.initialize() 的 instructions 字段填充（见 _session_runner）。
        self.description = ""
        self._capabilities_cache: list[Capability] | None = None
        self._closed = False
        # ── session owner task 状态 ──────────────────────────────────────────
        self._session: ClientSession | None = None
        self._runner: asyncio.Task | None = None
        self._ready: asyncio.Event | None = None
        self._stop: asyncio.Event | None = None
        # ── 熔断/退避状态 ────────────────────────────────────────────────────
        self._backoff_sec = reconnect_base_delay_sec
        self._cooldown_until = 0.0  # loop time；< now 表示可再尝试
        self._consecutive_failures = 0

    # ── 连接生命周期（非阻塞） ─────────────────────────────────────────────

    async def _ensure_connected(self) -> None:
        """非阻塞：已连上则直接返回；否则在后台起一次连接尝试（受冷却限制）后立即返回。

        调用方（list/invoke）据 `self._session` 是否就绪自行降级，绝不在此 await 握手。
        """
        if self._closed:
            raise RuntimeError(f"MCP provider '{self._cfg.name}' has been closed")
        if self._session is not None:
            return
        if self._runner is not None and not self._runner.done():
            return  # 正在连接
        if self._loop_time() < self._cooldown_until:
            return  # 熔断冷却中
        self._start_runner()

    def _start_runner(self) -> None:
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._capabilities_cache = None
        self._runner = asyncio.create_task(self._session_runner(), name=f"{self.name}-session")

    async def start(self) -> bool:
        """服务启动时预连接（非 agent 路径，可阻塞至 connect_timeout）。

        非致命：失败仅记日志并进入退避，agent 路径稍后惰性重试。返回是否连上。
        """
        if self._closed:
            return False
        if self._session is not None:
            return True
        if self._runner is None or self._runner.done():
            self._start_runner()
        assert self._ready is not None
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=self._cfg.connect_timeout_sec + 5)
        except TimeoutError:
            return False
        return self._session is not None

    async def _session_runner(self) -> None:
        """owner task：进入 transport + ClientSession，initialize，发布 session，等待 stop。

        进入/退出都在本 task 内完成（退出由 _stop 触发或被 cancel），满足 anyio cancel
        scope 的同 task 约束。一次尝试：成功则服务到断开/停止；失败/断开则记退避后退出。
        """
        from mcp import ClientSession

        timeout = (
            timedelta(seconds=self._cfg.timeout_per_call_sec)
            if self._cfg.timeout_per_call_sec else None
        )
        try:
            async with self._transport_cm() as transport:
                read_stream, write_stream = transport[0], transport[1]
                async with ClientSession(
                    read_stream, write_stream,
                    read_timeout_seconds=timeout,
                    client_info=self._client_info(),
                    message_handler=self._on_server_message,
                ) as session:
                    # 不用 asyncio.wait_for 包 initialize（会把 anyio 会话协程 reparent 到新
                    # task 导致死锁）。连接超时由 transport 自身控制（http: streamablehttp_client
                    # 的 timeout；stdio: 进程启动），握手响应由 ClientSession 的读超时兜底。
                    init_result = await session.initialize()
                    # MCP `instructions`：描述如何使用该 server 及其工具，用作 provider description。
                    self.description = init_result.instructions or ""
                    self._session = session
                    self._on_connected()
                    if self._ready is not None:
                        self._ready.set()  # 连上即放行 start() 的等待
                    assert self._stop is not None
                    await self._stop.wait()
        except asyncio.CancelledError:
            pass  # 主动 close/teardown，不算失败
        except BaseException as exc:
            self._on_failure(exc)
        finally:
            self._session = None
            if self._ready is not None:
                self._ready.set()  # 解除 start() 的等待（即使连接失败）

    async def _on_server_message(self, message: Any) -> None:
        """ClientSession message_handler：监听 server 推送。

        会话存活期间收到 `notifications/tools/list_changed` 时使工具快照失效——下一次
        `list()` 会重新拉取，从而反映 server 动态增减的工具。其它消息/异常忽略。
        """
        from mcp import types as t
        if isinstance(message, t.ServerNotification) and isinstance(
            message.root, t.ToolListChangedNotification
        ):
            self._capabilities_cache = None
            logger.info("MCP '%s' announced tools/list_changed; tool cache invalidated", self.name)

    def _on_connected(self) -> None:
        if self._consecutive_failures:
            logger.info("MCP '%s' connected (after %d failed attempt(s))",
                        self.name, self._consecutive_failures)
        self._consecutive_failures = 0
        self._backoff_sec = self._reconnect_base_delay_sec
        self._cooldown_until = 0.0

    def _on_failure(self, exc: BaseException) -> None:
        self._consecutive_failures += 1
        self._cooldown_until = self._loop_time() + self._backoff_sec
        reason = _short_error(exc)
        # 首次失败 WARNING；后续连续失败降级到 DEBUG，避免日志刷屏。
        if self._consecutive_failures == 1:
            logger.warning(
                "MCP '%s' unavailable (%s); backing off %.0fs before next attempt",
                self.name, reason, self._backoff_sec,
            )
        else:
            logger.debug(
                "MCP '%s' still unavailable (%s; attempt %d); backoff %.0fs",
                self.name, reason, self._consecutive_failures, self._backoff_sec,
            )
        self._backoff_sec = min(self._backoff_sec * 2, self._max_backoff_sec)

    @staticmethod
    def _loop_time() -> float:
        try:
            return asyncio.get_running_loop().time()
        except RuntimeError:
            return 0.0

    def _client_info(self) -> mcp_types.Implementation:
        from mcp import types as t
        return t.Implementation(name=self._client_name, version=self._client_version)

    def _transport_cm(self) -> Any:
        """返回 transport 的 async context manager（yield (read, write, ...)）。

        测试可覆写此方法以接入内存 server。
        """
        if self._cfg.transport == "stdio":
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client
            if not self._cfg.command:
                raise RuntimeError(f"MCP '{self._cfg.name}': stdio transport requires 'command'")
            return stdio_client(StdioServerParameters(
                command=self._cfg.command[0],
                args=list(self._cfg.command[1:]),
                env={**os.environ, **self._cfg.env} if self._cfg.env else None,
            ))
        from mcp.client.streamable_http import streamable_http_client
        http_client = httpx.AsyncClient(
            base_url=self._cfg.url,
            headers=self._cfg.headers,
            timeout=self._cfg.timeout_per_call_sec,
            trust_env=self._cfg.trust_env,
            verify=self._cfg.ssl_verify
        )
        return streamable_http_client(
            url=self._cfg.url,
            http_client=http_client
        )

    async def _teardown_runner(self) -> None:
        """触发 owner task 退出（或 cancel），等其 settle。"""
        self._session = None
        if self._stop is not None:
            self._stop.set()
        runner, self._runner = self._runner, None
        if runner is None:
            return
        try:
            # wait_for 超时会自动 cancel runner，并在 runner task 内完成 CM 退出。
            await asyncio.wait_for(runner, timeout=5.0)
        except (TimeoutError, asyncio.CancelledError):
            pass
        except Exception:
            pass

    # ── ToolCapabilityProvider 接口 ────────────────────────────────────────

    async def list(self, ctx: ProviderContext) -> list[Capability]:
        if self._capabilities_cache is not None:
            return self._capabilities_cache
        await self._ensure_connected()
        if self._session is None:
            return []  # 后台连接中 / 冷却中 —— 优雅降级，本轮无该 server 工具
        try:
            tools = await self._list_tools_all()
            caps: list[Capability] = []
            for t in tools:
                cap_id = f"mcp:{self._cfg.name}:{t.name}"
                purposes = self._cfg.capability_purpose_override.get(t.name, self._cfg.default_purposes)
                caps.append(ToolCapability(
                    id=cap_id, name=t.name, kind="tool",
                    purposes=purposes,  # type: ignore[arg-type]
                    description=t.description or "",
                    input_schema=t.inputSchema or {},
                    side_effects=any(kw in t.name.lower() for kw in _SIDE_EFFECT_KEYWORDS),
                ))
            self._capabilities_cache = caps
            return caps
        except Exception as e:
            logger.error("MCP '%s' list() failed: %s", self._cfg.name, e)
            return []

    async def _list_tools_all(self) -> list[mcp_types.Tool]:
        assert self._session is not None
        tools: list[mcp_types.Tool] = []
        cursor: str | None = None
        while True:
            result = await self._session.list_tools(cursor=cursor)
            tools.extend(result.tools)
            cursor = result.nextCursor
            if not cursor:
                break
        return tools

    def invoke(self, capability_id: str, arguments: dict[str, Any], ctx: ProviderContext) -> AsyncIterator[CapabilityEvent]:
        return self._invoke_impl(capability_id, arguments)

    async def _invoke_impl(self, capability_id: str, arguments: dict[str, Any]) -> AsyncIterator[CapabilityEvent]:
        await self._ensure_connected()
        if self._session is None:
            yield CapabilityEvent(kind="error", payload={
                "code": "NOT_CONNECTED",
                "message": f"MCP server '{self._cfg.name}' is not connected (status={self.connection_status})",
            })
            return
        tool_name = capability_id.rsplit(":", 1)[-1]
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments=arguments),
                timeout=self._cfg.timeout_per_call_sec,
            )
            text, images = _parse_tool_result(result)
            # 图片直接进 content（`str | list[ContentPart]`，与三个执行入口同一个联合
            # 类型）。拆分、校验、外部化都归 gateway，provider 不必认识 blob store。
            content: "str | list[ContentPart]" = text
            if images:
                content = ([TextPart(text=text), *images] if text else list(images))
            yield CapabilityEvent(
                kind="result",
                payload={"content": content, "metadata": {"is_error": bool(result.isError)}},
            )
        except TimeoutError:
            yield CapabilityEvent(kind="error", payload={"code": "TIMEOUT", "message": "MCP call timed out"})
        except Exception as e:
            yield CapabilityEvent(kind="error", payload={"code": "INVOKE_ERROR", "message": str(e)})

    async def cancel(self, invocation_id: str, ctx: ProviderContext) -> None:
        pass

    async def close(self) -> None:
        """Stop the session owner task and release the transport."""
        self._closed = True
        await self._teardown_runner()

    @property
    def connection_status(self) -> str:
        if self._closed:
            return "DISCONNECTED"
        if self._session is not None:
            return "CONNECTED"
        if self._runner is not None and not self._runner.done():
            return "CONNECTING"
        if self._loop_time() < self._cooldown_until:
            return "COOLDOWN"
        return "DISCONNECTED"

    async def describe(self, ctx: ProviderContext) -> CapabilityProviderInfo:
        return CapabilityProviderInfo(
            name=self.name,
            capability_count=len(self._capabilities_cache or []),
            supports_streaming=False,
            supports_cancel=False,
            description=self.description,
        )


def _parse_tool_result(
    result: mcp_types.CallToolResult,
) -> "tuple[str, list[ImagePart]]":
    """从 CallToolResult 抽出 ``(文本, 图片 part)``。

    **文本口径逐字节不变**：优先 text content，退回 structuredContent JSON，都没有则
    空串——图片的有无不参与这个判断（一个 server 同时给 structuredContent 与图是合法的，
    两样都该到模型手上）。

    ``ImageContent``（``type == "image"``，字段 ``data`` = base64、``mimeType``）改造前
    在这里被**静默丢弃**：只收 ``item.text``，图片连一条占位都不留，模型不知道自己少拿
    了东西。现在映射成 ``ImagePart``，由 `_invoke_impl` 放进 result 的 ``content``
    （`str | list[ContentPart]`）交给 gateway。

    ``mimeType`` **原样带过去、不在这里归一**：白名单匹配那侧
    （`core.utils.content._normalize_media_type`）自己会处理大小写 / ``; charset=`` /
    ``image/jpg`` 这些变体，而 wire 序列化要的是 server 报的原值。类型不合规、base64
    畸形、超 5 MiB 都由 gateway 的 `legalize_tool_result_parts` 换成占位——本函数不校验、
    不外部化、也不抛（MCP server 是外部输入，畸形是常态而非异常）。

    ``EmbeddedResource`` / ``AudioContent`` 暂不处理（音视频不在本期范围，见多模态设计
    §9）：它们既无 ``.text`` 也非 ``type == "image"``，落到最后被跳过，与改造前同。
    """
    texts: list[str] = []
    images: list[ImagePart] = []
    for item in (result.content or []):
        text_field = getattr(item, "text", None)
        if text_field:
            texts.append(text_field)
            continue
        # 全程 getattr 而非属性访问：`result.content` 是五个 content 类型的联合
        # （TextContent / ImageContent / AudioContent / ResourceLink / EmbeddedResource），
        # 按 `type` 取值判分支不会让类型收窄，直接写 `item.data` 是 union-attr 错。
        data = getattr(item, "data", None)
        if getattr(item, "type", "") == "image" and data:
            images.append(ImagePart(
                data=data,
                media_type=getattr(item, "mimeType", "") or "",
                source_type="base64",
            ))
    if texts:
        return "\n".join(texts), images
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False), images
    return "", images


def _short_error(exc: BaseException) -> str:
    """把 anyio/3.11 ExceptionGroup 解包到首个叶子异常，给出可读原因。"""
    seen = 0
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions and seen < 5:
        exc = exc.exceptions[0]
        seen += 1
    return f"{type(exc).__name__}: {exc}"
