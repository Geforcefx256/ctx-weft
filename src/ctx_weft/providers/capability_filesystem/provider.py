"""FilesystemToolsProvider：文件系统操作工具集 + per-session workspace 管理。

工具（shell / read_file / write_file / edit_file / glob / grep）都在 session 的工作目录下运作：
  - shell 以 workspace 为 cwd；
  - read_file / write_file / edit_file / glob / grep 把相对路径锚定到 workspace。

workspace 是本 provider 的内部概念，core 不知道它的存在：host 在 session 启动前调用
register_session(session_id, 绝对路径) 登记，本 provider 维护 session_id → workspace 映射。
对 core 而言本 provider 只实现两个协议：
  - SpillSink（spill）：CapabilityGateway 截断超长输出时落盘到对应 workspace；
  - SessionScopedCapabilityProvider（deregister_session）：session 结束时释放映射。
register_session / workspace_for 是 provider 自有方法（host 接线用），不在任何协议内。
"""

from __future__ import annotations

import asyncio
import dataclasses
import glob as _glob
import logging
import os
import platform
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols.capability import (
    Capability,
    CapabilityEvent,
    CapabilityProviderInfo,
    SessionScopedCapabilityProvider,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.context import ProviderContext
from ctx_weft.protocols.filesystem import FS_PROVIDER_NAME, SpillSink
from ctx_weft.providers._encoding import decode_console
from ctx_weft.providers._script_runner import run_with_liveness
from ctx_weft.providers._tooldecl import make_tool_registry
from ctx_weft.providers.capability_filesystem._bash_safety import (
    BASH_BLACKLIST,
    check_command_safety,
)
from ctx_weft.providers.capability_filesystem._file_reader import (
    ReadConfig,
    read_byte_window,
    read_lines,
)
from ctx_weft.providers.capability_filesystem._grep import GrepConfig, run_grep
from ctx_weft.providers.capability_filesystem._venv import (
    VenvError,
    command_is_python,
    ensure_venv,
    venv_env,
    venv_layout,
)

logger = logging.getLogger(__name__)

tool, _FS_TOOLS, _FS_IMPLS = make_tool_registry(FS_PROVIDER_NAME)

# ── 共享常量 ──────────────────────────────────────────────────────────────────

_BASH_IDLE_TIMEOUT_SEC_DEFAULT = 30
_BASH_HARD_CAP_SEC_DEFAULT = 3600
_BASH_MAX_OUTPUT_BYTES_DEFAULT = 50_000
_FILE_READ_DEFAULT_LINES = 500
_FILE_READ_MAX_BYTES = 20_480
_FILE_READ_MAX_LINE_BYTES = 4096
_FILE_READ_COUNT_MAX_BYTES = 5_242_880
_GLOB_MAX_RESULTS_DEFAULT = 500
_GREP_MAX_RESULTS_DEFAULT = 1000


def _shell_description() -> str:
    """Build a platform-aware description so the LLM picks the right shell syntax.

    The description states the concrete host OS, which shell the command runs in,
    which command family is available vs unavailable for that OS, and the list of
    commands that are blocked regardless of OS (``BASH_BLACKLIST``).
    """
    system = platform.system()  # 'Windows' | 'Linux' | 'Darwin'
    detail = platform.platform()
    blocked = ", ".join(sorted(BASH_BLACKLIST))
    # Steer the model away from using the shell as a general-purpose file tool.
    steer = (
        "Run a program or command (build/test/lint, git, package managers, scripts) "
        "and return its stdout/stderr. Do not use this tool to read, write, edit, or "
        "search files — use read_file / write_file / edit_file / glob / grep for that. "
    )
    blocked_note = (
        f"The following commands are blocked on every OS and will be rejected: {blocked}."
    )
    venv_note = (
        " python/pip run inside a .venv that is created automatically in the working "
        "directory on first use, so installs and runs stay isolated and reproducible."
    )
    blocked_note = blocked_note + venv_note
    if system == "Windows":
        return (
            f"{steer}"
            f"Host OS is Windows ({detail}); commands run via cmd.exe. "
            f"Use Windows commands (e.g. dir, type, copy, findstr, where); "
            f"Linux/Unix commands such as ls, cat, grep are NOT available. "
            f"Paths: write 'D:\\foo\\bar' (single backslash) or 'D:/foo/bar' "
            f"(forward slashes) — do NOT double-escape backslashes. "
            f"When you need PowerShell, call "
            f"powershell -NoProfile -Command \"...\"; on Windows PowerShell 5.1 "
            f"prepend [Console]::OutputEncoding=[Text.UTF8Encoding]::new(); so "
            f"Chinese output is not garbled (PowerShell 7 already uses UTF-8). "
            f"{blocked_note}"
        )
    shell_kind = "macOS" if system == "Darwin" else "Linux"
    return (
        f"{steer}"
        f"Host OS is {shell_kind} ({detail}); commands run via /bin/sh. "
        f"Use POSIX shell commands (e.g. ls, cat, grep, find); "
        f"Windows-only commands such as dir, type, findstr are NOT available. "
        f"{blocked_note}"
    )


def _allowed_dirs(ctx: ProviderContext | None) -> list[Path]:
    if ctx is None:
        return []
    raw = ctx.extra.get("allowed_dirs") or []
    return [d if isinstance(d, Path) else Path(d) for d in raw]


def _workspace(ctx: ProviderContext | None) -> Path | None:
    """该次调用的 session 工作目录（provider.invoke 注入到 ctx.extra）。"""
    if ctx is None:
        return None
    ws = ctx.extra.get("workspace")
    return Path(ws) if ws else None


def _check_path(path: Path, allowed_dirs: list[Path]) -> bool:
    """Return True if path is within at least one allowed directory, or no restriction."""
    if not allowed_dirs:
        return True
    resolved = path.resolve()
    return any(str(resolved).startswith(str(d.resolve())) for d in allowed_dirs)


def _resolve(path: str, ctx: ProviderContext | None) -> Path:
    """相对路径锚定到 session workspace；绝对路径原样。"""
    p = Path(path)
    if p.is_absolute():
        return p
    ws = _workspace(ctx)
    return (ws / p) if ws else p


# ── 工具实现 ──────────────────────────────────────────────────────────────────


def _read_config(ctx: ProviderContext | None) -> ReadConfig:
    e = ctx.extra if ctx else {}
    return ReadConfig(
        default_lines=e.get("file_read_default_lines") or _FILE_READ_DEFAULT_LINES,
        max_bytes=e.get("file_read_max_bytes") or _FILE_READ_MAX_BYTES,
        max_line_bytes=e.get("file_read_max_line_bytes") or _FILE_READ_MAX_LINE_BYTES,
        count_max_bytes=e.get("file_read_count_max_bytes") or _FILE_READ_COUNT_MAX_BYTES,
    )


@tool(purposes=["act"], side_effects=True, description=_shell_description())
async def shell(
    command: Annotated[str, "Shell command to execute"],
    *,
    ctx: ProviderContext | None = None,
) -> AsyncIterator[CapabilityEvent]:
    """Execute a shell command and return stdout/stderr."""
    if not command.strip():
        yield CapabilityEvent(kind="error", payload={"code": "EMPTY_COMMAND", "message": "command is required"})
        return

    _blacklist = ctx.extra.get("bash_blacklist") if ctx else None
    err = check_command_safety(command, _blacklist) if _blacklist is not None else check_command_safety(command)
    if err:
        yield CapabilityEvent(
            kind="error",
            payload={"code": "COMMAND_BLACKLISTED", "message": err},
        )
        return

    yield CapabilityEvent(kind="progress", payload={"status": "starting", "command": command})

    ws = _workspace(ctx)
    cwd = str(ws) if ws else None
    idle = (ctx.extra.get("bash_idle_timeout_sec") if ctx else None) or _BASH_IDLE_TIMEOUT_SEC_DEFAULT
    hard = (ctx.extra.get("bash_hard_cap_sec") if ctx else None) or _BASH_HARD_CAP_SEC_DEFAULT
    max_out = (ctx.extra.get("bash_max_output_bytes") if ctx else None) or _BASH_MAX_OUTPUT_BYTES_DEFAULT

    logger.info("shell command (repr): %r", command)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    # 调用方(如 skill 委托执行)注入的额外环境变量(如 SKILL_DIR)。venv_env 之后会保留这些键。
    _extra_env = ctx.extra.get("extra_env") if ctx else None
    if isinstance(_extra_env, dict) and _extra_env:
        env.update(_extra_env)

    # Python .venv 引导：检测到 python/pip 类命令时在 workspace 下懒建并「激活」.venv。
    # 每次 bash_exec 是全新子进程，故只能在子进程 env 层注入（PATH/VIRTUAL_ENV）。
    auto_venv = ctx.extra.get("bash_auto_venv", True) if ctx else True
    venv_dir = (ctx.extra.get("bash_venv_dir") if ctx else None) or ".venv"
    venv_python = ctx.extra.get("bash_venv_python") if ctx else None
    if auto_venv and ws and command_is_python(command):
        venv_path, _, python_exe = venv_layout(ws, venv_dir)
        if not python_exe.exists():
            yield CapabilityEvent(
                kind="progress",
                payload={"status": "creating_venv", "path": str(venv_path)},
            )
        try:
            await ensure_venv(ws, venv_dir, creator_python=venv_python)
        except VenvError as e:
            yield CapabilityEvent(
                kind="error",
                payload={"code": "VENV_ERROR", "message": str(e)},
            )
            return
        env = venv_env(env, venv_path)

    # The runner streams decoded output via on_output; bridge those callbacks
    # through a queue so this async generator can yield stdout events live.
    queue: asyncio.Queue = asyncio.Queue()

    def _on_output(_stream: str, text: str) -> None:
        queue.put_nowait(("out", text))

    async def _run() -> None:
        try:
            result = await run_with_liveness(
                command, cwd=cwd, env=env,
                idle_timeout_sec=idle, hard_cap_sec=hard, output_limit_bytes=max_out,
                on_output=_on_output,
            )
            queue.put_nowait(("done", result))
        except Exception as e:  # noqa: BLE001 — surfaced as an error event below
            queue.put_nowait(("exc", e))

    runner = asyncio.create_task(_run())
    try:
        while True:
            kind, payload = await queue.get()
            if kind == "out":
                yield CapabilityEvent(kind="stdout", payload={"data": payload})
            elif kind == "exc":
                logger.exception("shell failed: %s", command)
                yield CapabilityEvent(kind="error", payload={"code": "EXEC_ERROR", "message": str(payload)})
                return
            elif kind == "done":
                result = payload
                if result.timed_out:
                    survivor = "" if result.terminated_clean else (
                        f" WARNING: {len(result.survivors)} process(es) may still be running."
                    )
                    yield CapabilityEvent(kind="error", payload={
                        "code": "TIMEOUT",
                        "message": f"Command timed out ({result.timeout_kind}).{survivor}",
                    })
                    return
                exit_code = result.exit_code or 0
                # Fold stderr into content to preserve the old stderr=STDOUT merge.
                content = result.stdout + result.stderr
                yield CapabilityEvent(kind="result", payload={
                    "content": content,
                    "metadata": {"exit_code": exit_code, "is_error": exit_code != 0},
                })
                return
    finally:
        if not runner.done():
            runner.cancel()


@tool(purposes=["act", "compact"], side_effects=False, spillable=False)
async def read_file(
    path: Annotated[str, "File path; relative paths are resolved against the workspace"],
    offset: Annotated[int | None, "1-based start line (line mode; default 1)"] = None,
    limit: Annotated[int | None, "Number of lines to read (line mode; default 500)"] = None,
    *,
    byte_offset: Annotated[int | None, "Raw byte offset to start at (byte mode; escape hatch for oversized single lines)"] = None,
    byte_limit: Annotated[int | None, "Max bytes to read in byte mode (clamped to the per-call budget)"] = None,
    ctx: ProviderContext | None = None,
) -> AsyncIterator[CapabilityEvent]:
    """Read a plain-text file, with line numbers, one page at a time.
    Plain-text only (code, .txt, .md, .json, .csv, config). Does NOT decode
    binary/document formats (.pptx, .docx, .xlsx, .pdf, images, etc.) — those
    yield garbled bytes; use a format-specific tool or convert to text first.
    Line mode (default): reads `limit` lines from line `offset` (1-based). Large
    files paginate — when the result ends with a "Continue:" hint, call again
    with the suggested `offset` for the next page; don't try to read it all at once.
    Byte mode: `byte_offset` (+ optional `byte_limit`) reads raw bytes without
    line numbers. Escape hatch for when line mode reports a truncated line: page
    `byte_offset` up to the reported `byte_end`, then resume line mode at the next line.
    `offset`/`limit` and `byte_offset` are mutually exclusive.
    """
    if not path:
        yield CapabilityEvent(kind="error", payload={"code": "MISSING_PATH", "message": "path is required"})
        return

    if byte_offset is not None:
        if offset is not None or limit is not None:
            yield CapabilityEvent(kind="error", payload={
                "code": "INVALID_ARGS",
                "message": "specify either line offset or byte_offset, not both",
            })
            return
        if byte_offset < 0:
            yield CapabilityEvent(kind="error", payload={
                "code": "INVALID_ARGS", "message": "byte_offset must be >= 0"})
            return
        if byte_limit is not None and byte_limit < 1:
            yield CapabilityEvent(kind="error", payload={
                "code": "INVALID_ARGS", "message": "byte_limit must be >= 1"})
            return
    else:
        if offset is not None and offset < 1:
            yield CapabilityEvent(kind="error", payload={
                "code": "INVALID_ARGS", "message": "offset is 1-based; must be >= 1"})
            return
        if limit is not None and limit < 1:
            yield CapabilityEvent(kind="error", payload={
                "code": "INVALID_ARGS", "message": "limit must be >= 1"})
            return

    file_path = _resolve(path, ctx)
    if not _check_path(file_path, _allowed_dirs(ctx)):
        yield CapabilityEvent(kind="error", payload={
            "code": "PATH_NOT_ALLOWED", "message": "Path is outside allowed directories"})
        return

    try:
        if not file_path.exists() or not file_path.is_file():
            yield CapabilityEvent(kind="error", payload={
                "code": "FILE_NOT_FOUND", "message": f"File not found: {file_path}"})
            return

        cfg = _read_config(ctx)
        if byte_offset is not None:
            result = await asyncio.to_thread(read_byte_window, file_path, byte_offset, byte_limit, cfg)
        else:
            result = await asyncio.to_thread(read_lines, file_path, offset, limit, cfg)

        yield CapabilityEvent(kind="result", payload={
            "content": result.content,
            "metadata": result.metadata,
        })
    except Exception as e:
        logger.exception("read_file failed: %s", file_path)
        yield CapabilityEvent(kind="error", payload={"code": "READ_ERROR", "message": str(e)})


@tool(purposes=["act"], side_effects=True)
async def write_file(
    path: Annotated[str, "File path (relative resolved against workspace; parents auto-created)"],
    content: Annotated[str, "Text content to write"],
    *,
    ctx: ProviderContext | None = None,
) -> AsyncIterator[CapabilityEvent]:
    """Create a file, or overwrite it entirely if it already exists.
    Parent directories are auto-created; to change part of an existing file,
    use edit_file instead.
    Short files and small rewrites: write them in one call, that is what this
    tool is for. For a long file — roughly over ~200 lines, or several
    independent logical blocks — write it in stages instead: first a compact
    skeleton (signatures, section headers, placeholder comments), then fill in
    one block at a time with edit_file, checking each edit landed before the
    next. One giant write is where output gets truncated or malformed, and
    recovering means redoing the whole file rather than one block.
    """
    if not path:
        yield CapabilityEvent(kind="error", payload={"code": "MISSING_PATH", "message": "path is required"})
        return

    file_path = _resolve(path, ctx)
    if not _check_path(file_path, _allowed_dirs(ctx)):
        yield CapabilityEvent(
            kind="error",
            payload={"code": "PATH_NOT_ALLOWED", "message": "Path is outside allowed directories"},
        )
        return

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        yield CapabilityEvent(
            kind="result",
            payload={
                "content": f"Written {len(content)} chars to {file_path}",
                "metadata": {"path": str(file_path), "size_chars": len(content)},
            },
        )
    except Exception as e:
        logger.exception("write_file failed: %s", file_path)
        yield CapabilityEvent(kind="error", payload={"code": "WRITE_ERROR", "message": str(e)})


@tool(purposes=["act"], side_effects=True)
async def edit_file(
    path: Annotated[str, "File path; relative paths are resolved against the workspace"],
    old_string: Annotated[str, "Exact text to find"],
    new_string: Annotated[str, "Replacement text"],
    *,
    replace_all: Annotated[bool, "Replace every occurrence instead of requiring a unique match"] = False,
    ctx: ProviderContext | None = None,
) -> AsyncIterator[CapabilityEvent]:
    """Replace an exact `old_string` with `new_string` in a file, in place.
    By default `old_string` must match exactly once — add surrounding context to
    make it unique, or pass `replace_all=true` to replace every occurrence.
    """
    if not path:
        yield CapabilityEvent(kind="error", payload={"code": "MISSING_PATH", "message": "path is required"})
        return
    if old_string == new_string:
        yield CapabilityEvent(kind="error", payload={
            "code": "INVALID_ARGS", "message": "old_string and new_string must differ"})
        return

    file_path = _resolve(path, ctx)
    if not _check_path(file_path, _allowed_dirs(ctx)):
        yield CapabilityEvent(
            kind="error",
            payload={"code": "PATH_NOT_ALLOWED", "message": "Path is outside allowed directories"},
        )
        return

    try:
        if not file_path.exists() or not file_path.is_file():
            yield CapabilityEvent(kind="error", payload={
                "code": "FILE_NOT_FOUND", "message": f"File not found: {file_path}"})
            return

        text = file_path.read_text(encoding="utf-8")
        count = text.count(old_string)
        if count == 0:
            yield CapabilityEvent(kind="error", payload={
                "code": "STRING_NOT_FOUND", "message": "old_string not found in file"})
            return
        if count > 1 and not replace_all:
            yield CapabilityEvent(kind="error", payload={
                "code": "NOT_UNIQUE",
                "message": (
                    f"old_string matches {count} times; add surrounding context to make it "
                    f"unique, or pass replace_all=true"
                ),
            })
            return

        replacements = count if replace_all else 1
        new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        file_path.write_text(new_text, encoding="utf-8")
        yield CapabilityEvent(
            kind="result",
            payload={
                "content": f"Replaced {replacements} occurrence(s) in {file_path}",
                "metadata": {"path": str(file_path), "replacements": replacements, "replace_all": replace_all},
            },
        )
    except Exception as e:
        logger.exception("edit_file failed: %s", file_path)
        yield CapabilityEvent(kind="error", payload={"code": "EDIT_ERROR", "message": str(e)})


@tool(purposes=["act", "compact"], side_effects=False)
async def glob(
    pattern: Annotated[str, "Glob pattern relative to base_dir, e.g. '**/*.py'"],
    base_dir: Annotated[str, "Base directory (relative resolved against workspace)"] = ".",
    *,
    ctx: ProviderContext | None = None,
) -> AsyncIterator[CapabilityEvent]:
    """Find files by name/path pattern (e.g. '**/*.py').
    To search file *contents* instead, use grep.
    """
    if not pattern:
        yield CapabilityEvent(kind="error", payload={"code": "MISSING_PATTERN", "message": "pattern is required"})
        return

    try:
        _max_glob = (ctx.extra.get("glob_max_results") if ctx else None) or _GLOB_MAX_RESULTS_DEFAULT
        root = _resolve(base_dir, ctx)
        matches = _glob.glob(str(root / pattern), recursive=True)[:_max_glob]
        yield CapabilityEvent(
            kind="result",
            payload={
                "content": "\n".join(matches) if matches else "(no matches)",
                "metadata": {
                    "pattern": pattern,
                    "base_dir": str(root),
                    "count": len(matches),
                    "truncated": len(matches) == _max_glob,
                },
            },
        )
    except Exception as e:
        logger.exception("glob failed: %s", pattern)
        yield CapabilityEvent(kind="error", payload={"code": "GLOB_ERROR", "message": str(e)})


@tool(purposes=["act", "compact"], side_effects=False)
async def grep(
    pattern: Annotated[str, "Regular expression to search for in file contents"],
    path: Annotated[str, "File or directory to search; relative paths resolved against the workspace"] = ".",
    file_glob: Annotated[str | None, "Glob to filter which files are searched, e.g. '*.py'"] = None,
    output_mode: Annotated[str, "'files_with_matches' (default, just paths) or 'content' (matching lines)"] = "files_with_matches",
    *,
    ignore_case: Annotated[bool, "Case-insensitive match"] = False,
    before_context: Annotated[int | None, "Lines of context before each match (content mode)"] = None,
    after_context: Annotated[int | None, "Lines of context after each match (content mode)"] = None,
    context: Annotated[int | None, "Lines of context on both sides (content mode); overrides before/after"] = None,
    ctx: ProviderContext | None = None,
) -> AsyncIterator[CapabilityEvent]:
    """Search file contents by regular expression.
    Uses ripgrep when available (skips .gitignored/binary files), else a built-in
    Python walk (skips non-UTF-8 files). The chosen backend is reported in
    `metadata.backend`. `output_mode='content'` returns `path:line:text` lines;
    `before_context`/`after_context`/`context` add neighbor lines in that mode.
    """
    if not pattern:
        yield CapabilityEvent(kind="error", payload={"code": "MISSING_PATTERN", "message": "pattern is required"})
        return
    if output_mode not in ("files_with_matches", "content"):
        yield CapabilityEvent(kind="error", payload={
            "code": "INVALID_ARGS",
            "message": "output_mode must be 'files_with_matches' or 'content'",
        })
        return

    root = _resolve(path, ctx)
    if not _check_path(root, _allowed_dirs(ctx)):
        yield CapabilityEvent(kind="error", payload={
            "code": "PATH_NOT_ALLOWED", "message": "Path is outside allowed directories"})
        return
    if not root.exists():
        yield CapabilityEvent(kind="error", payload={
            "code": "PATH_NOT_FOUND", "message": f"Path not found: {root}"})
        return

    before = context if context is not None else (before_context or 0)
    after = context if context is not None else (after_context or 0)
    max_results = (ctx.extra.get("grep_max_results") if ctx else None) or _GREP_MAX_RESULTS_DEFAULT

    try:
        result = await asyncio.to_thread(
            run_grep, pattern, root,
            file_glob=file_glob, output_mode=output_mode, ignore_case=ignore_case,
            before=before, after=after, cfg=GrepConfig(max_results=max_results),
        )
        yield CapabilityEvent(kind="result", payload={
            "content": result.content, "metadata": result.metadata})
    except ValueError as e:  # un-compilable regex (Python backend)
        yield CapabilityEvent(kind="error", payload={"code": "INVALID_ARGS", "message": str(e)})
    except Exception as e:
        logger.exception("grep failed: %s", pattern)
        yield CapabilityEvent(kind="error", payload={"code": "GREP_ERROR", "message": str(e)})


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class FilesystemConfig:
    """Provider 运行时配置。权限控制由 auth 层负责，不在此处理。"""
    allowed_dirs: list[Path] = field(default_factory=list)
    bash_idle_timeout_sec: int = 30
    bash_hard_cap_sec: int = 3600
    bash_max_output_bytes: int = 50_000
    bash_auto_venv: bool = True
    bash_venv_dir: str = ".venv"
    bash_venv_python: str | None = None  # 创建 venv 用的真 Python；None=回退 sys.executable
    bash_blacklist: frozenset[str] | None = None  # host 收窄硬黑名单；None=用 _bash_safety 默认集
    file_read_default_lines: int = 500
    file_read_max_bytes: int = 20_480
    file_read_max_line_bytes: int = 4096
    file_read_count_max_bytes: int = 5_242_880
    glob_max_results: int = 500
    grep_max_results: int = 1000


# ── Provider ──────────────────────────────────────────────────────────────────


class FilesystemToolsProvider(ToolCapabilityProvider, SpillSink,
                              SessionScopedCapabilityProvider):
    """文件系统工具 provider：shell, read_file, write_file, edit_file, glob, grep + per-session workspace。

    实现三个面向 core 的契约：ToolCapabilityProvider（invoke 六个工具）、SpillSink（spill 落盘）、
    SessionScopedCapabilityProvider（deregister_session 清理）。register_session / workspace_for
    是 host 接线用的自有方法，不属于任何协议——workspace 对 core 不可见。
    """

    name = FS_PROVIDER_NAME

    # Rendered as the header paragraph for this whole tool group in the LLM prompt
    # (composer 把它写在 "#### filesystem tools" 标题下)。用来给整组工具定策略：
    # 文件读写/查找有专用工具，别拿 shell 当万能文件工具。
    description = (
        "Tools scoped to the session workspace; relative paths resolve against it. "
        "For files, use these rather than shell: read_file, write_file, edit_file, "
        "glob (find by name), grep (search contents). Use shell only to run programs "
        "— build/test/lint, git, package managers, scripts."
    )

    def __init__(self, config: FilesystemConfig | None = None) -> None:
        self._cfg = config or FilesystemConfig()
        self._invokers = self._build_invokers()
        self._workspaces: dict[str, str] = {}  # session_id → 绝对路径

    def _build_invokers(self) -> dict[str, Callable]:
        invokers = {
            name: (lambda f: lambda args, ctx: f(**args, ctx=ctx))(fn)
            for name, fn in _FS_IMPLS.items()
        }
        return invokers

    # ── workspace 生命周期 ────────────────────────────────────────────────────

    def register_session(self, session_id: str, workspace: str) -> None:
        """登记某 session 的工作目录（必须绝对路径）。host 接线用的自有方法，不在协议内。"""
        if not workspace or not os.path.isabs(workspace):
            raise ValueError(
                f"workspace must be an absolute path, got: {workspace!r}"
            )
        path = os.path.normpath(workspace)
        os.makedirs(path, exist_ok=True)
        self._workspaces[session_id] = path
        logger.info("Filesystem: registered workspace for session %s → %s", session_id, path)

    def deregister_session(self, session_id: str) -> None:
        """SessionScopedCapabilityProvider：会话被**逐出内存**时由 core 调用，释放映射。

        触发点是 `forget_session` / `purge_session`，**不是「会话跑完」**（2026-09-08
        生命周期改造前是后者）。所以宿主在每条执行入口之前都要确保重新登记——映射是
        内存缓存，真值住在宿主自己的存储里（本 provider 不认识那个存储，也不该认识）。
        """
        self._workspaces.pop(session_id, None)

    def workspace_for(self, ctx: ProviderContext) -> str | None:
        """该 session 已登记的工作目录绝对路径；未登记返回 None。host 接线用的自有方法。"""
        return self._workspaces.get(ctx.session_id)

    async def spill(self, content: str, ctx: ProviderContext, *, name_hint: str = "") -> str:
        """SpillSink：把超长内容落盘到该 session 的 workspace，返回**取回说明**；未登记则 raise。

        说明里点名 `fs__read_file`——落盘的文件就在 workspace 里，本 provider 自己的读取
        工具能翻页读回来。只给一条路径的话，模型得自己猜该用哪个工具、传什么参数。
        """
        ws = self.workspace_for(ctx)
        if ws is None:
            raise RuntimeError(f"no workspace registered for session {ctx.session_id!r}")
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (name_hint or "out"))
        file_path = Path(ws) / "tool_outputs" / f"{safe}_{generate_id('spill')}.txt"
        await asyncio.to_thread(self._write_text, file_path, content)
        return (f"Full text saved to {file_path} — read it with "
                f"{FS_PROVIDER_NAME}__read_file(path='{file_path}', offset=1, limit=500) "
                f"and page with the offset it suggests.")

    @staticmethod
    def _write_text(file_path: Path, content: str) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    # ── CapabilityProvider 接口 ───────────────────────────────────────────────

    async def list(self, ctx: ProviderContext) -> list[Capability]:
        return list(_FS_TOOLS.values())

    def invoke(
        self,
        capability_id: str,
        arguments: dict[str, Any],
        ctx: ProviderContext,
    ) -> AsyncIterator[CapabilityEvent]:
        # 注入本次 session 的 workspace + allowed_dirs + runtime limits，供工具读取
        extra = dict(ctx.extra)
        ws = self._workspaces.get(ctx.session_id)
        if ws:
            extra["workspace"] = ws
        if self._cfg.allowed_dirs:
            extra["allowed_dirs"] = self._cfg.allowed_dirs
        # 限额类:调用方(如 skill 委托)可经 ctx.extra 覆盖;未给才用 fs 配置。
        extra.setdefault("bash_idle_timeout_sec", self._cfg.bash_idle_timeout_sec)
        extra.setdefault("bash_hard_cap_sec", self._cfg.bash_hard_cap_sec)
        extra.setdefault("bash_max_output_bytes", self._cfg.bash_max_output_bytes)
        # 以下为 fs 强制(不可被调用方覆盖):workspace/venv 引导/读取与搜索限额。
        extra["bash_auto_venv"] = self._cfg.bash_auto_venv
        extra["bash_venv_dir"] = self._cfg.bash_venv_dir
        extra["bash_venv_python"] = self._cfg.bash_venv_python
        if self._cfg.bash_blacklist is not None:
            extra["bash_blacklist"] = self._cfg.bash_blacklist
        extra["file_read_default_lines"] = self._cfg.file_read_default_lines
        extra["file_read_max_bytes"] = self._cfg.file_read_max_bytes
        extra["file_read_max_line_bytes"] = self._cfg.file_read_max_line_bytes
        extra["file_read_count_max_bytes"] = self._cfg.file_read_count_max_bytes
        extra["glob_max_results"] = self._cfg.glob_max_results
        extra["grep_max_results"] = self._cfg.grep_max_results
        ctx = dataclasses.replace(ctx, extra=extra)
        return self._dispatch(capability_id, arguments, ctx)

    async def _dispatch(
        self,
        capability_id: str,
        arguments: dict[str, Any],
        ctx: ProviderContext,
    ) -> AsyncIterator[CapabilityEvent]:
        name = capability_id.split(":")[-1]
        invoker = self._invokers.get(name)
        if invoker is None:
            yield CapabilityEvent(
                kind="error",
                payload={"code": "UNKNOWN_CAPABILITY", "message": f"Unknown: {capability_id}"},
            )
            return
        async for ev in invoker(arguments, ctx):
            yield ev

    async def cancel(self, invocation_id: str, ctx: ProviderContext) -> None:
        pass

    async def describe(self, ctx: ProviderContext) -> CapabilityProviderInfo:
        return CapabilityProviderInfo(
            name=self.name,
            capability_count=len(_FS_TOOLS),
            supports_streaming=True,
            supports_cancel=False,
            description=self.description,
        )
