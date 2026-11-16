"""Base exception shared by public protocols and engine errors."""


class CtxWeftError(Exception):
    """所有 ctx-weft 异常的基类。"""

    code: str = "CTX_WEFT_ERROR"

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
