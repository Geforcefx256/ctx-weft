"""文件系统能力协议层。

把"文件系统操作"这一类 capability 从其他可选工具中分出来，只约定 core 真正依赖的两件事：

  1. 固定的工具名（FS_PROVIDER_NAME 前缀 + FsTool 常量），调用方与实现共用同一份。
  2. SpillSink：CapabilityGateway 截断超长工具输出时，把全文交出去并拿回一句**取回说明**
     ——core 因此不直接碰文件系统，也不需要知道「落到哪」「怎么取回来」。

注意：per-session workspace 的指定 / 登记 / 路径锚定**不在协议内**。那是 SpillSink 实现
（ctx_weft.providers.capability_filesystem.FilesystemToolsProvider）与 host 接线的细节：
host 在执行前把工作目录登记给具体 provider，core 全程只通过 SpillSink.spill() 与之交互。
session 结束时的清理走通用的 SessionScopedCapabilityProvider（见 protocols.capability）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ctx_weft.protocols.context import ProviderContext

FS_PROVIDER_NAME = "fs"


class FsTool:
    """文件系统工具的标准 capability id（前缀在协议层钉死，模板/授权按此引用）。"""

    SHELL = f"{FS_PROVIDER_NAME}:shell"
    READ_FILE = f"{FS_PROVIDER_NAME}:read_file"
    WRITE_FILE = f"{FS_PROVIDER_NAME}:write_file"
    GLOB = f"{FS_PROVIDER_NAME}:glob"


class SpillSink(ABC):
    """core 对「超长内容存到哪」的**唯一**契约：收下全文，返回一句**取回说明**。

    CapabilityGateway 在工具输出超阈值时调用 spill()，把 result 改为
    「截断提示 + 取回说明 + 头尾预览」。core 不直接碰文件系统、也不知道 workspace。

    返回值 SHALL 是一句**面向模型的、自足的**取回说明——写清楚**用哪个工具、传什么参数**
    才能读到全文，而不是一个裸引用。理由是歧义只有 sink 自己消得掉：core 拿到一个
    `/ws/tool_outputs/x.txt` 和一个 `results__read_tool_output(...)`，无从分辨前者要用
    `fs__read_file` 读、后者是个可照抄的调用形态；它若统一套一句「full text at {ref}」，
    对两种 sink 都不准确。写说明的成本只在**存进去的那一方**这里是零。

    句子形态（首字母大写、以句号收尾，gateway 原样嵌入，不再加任何框架词）：

        Full text saved to /ws/tool_outputs/out_abc.txt — read it with
        fs__read_file(path='/ws/tool_outputs/out_abc.txt', offset=1, limit=500).

    该 session 没有可落盘位置时 spill() 应 raise，调用方（gateway）据此回退到硬截断，
    并显式标注「不可取回」——**不留一个取不回来的引用**。

    实现见 capability_filesystem.FilesystemToolsProvider（落盘）与
    capability_results.ResultsCapabilityProvider（入内存 LRU + 自带回读工具）。
    """

    @abstractmethod
    async def spill(self, content: str, ctx: ProviderContext, *, name_hint: str = "") -> str:
        """收下 ``content``，返回一句面向模型的取回说明；存不下则 raise。"""
        ...

