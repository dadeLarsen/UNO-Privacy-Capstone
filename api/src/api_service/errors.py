"""Application errors with stable codes for the frontend."""


class ApiProblem(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable


class NotFound(ApiProblem):
    def __init__(self, what: str) -> None:
        super().__init__(status_code=404, code="not_found", message=f"{what} was not found or has expired.")


class ModelServiceError(ApiProblem):
    def __init__(self, *, status_code: int, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(status_code=status_code, code=code, message=message, retryable=retryable)


class GeneratorUnavailable(ApiProblem):
    def __init__(self, message: str) -> None:
        super().__init__(
            status_code=501,
            code="generator_not_available",
            message=message,
            retryable=False,
        )
