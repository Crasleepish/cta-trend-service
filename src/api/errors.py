from __future__ import annotations

from typing import Any, cast

from fastapi import Request
from fastapi.responses import JSONResponse

from .schemas import ApiError


class ApiErrorException(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


def error_response(_: Request, exc: Exception) -> JSONResponse:
    api_exc = cast(ApiErrorException, exc)
    payload = ApiError(
        code=api_exc.code, message=api_exc.message, details=api_exc.details
    ).model_dump()
    return JSONResponse(status_code=api_exc.status_code, content=payload)


def map_value_error(err: ValueError) -> ApiErrorException:
    msg = str(err)
    lowered = msg.lower()
    if "tilt" in lowered:
        return ApiErrorException("TILT_INVALID", msg, status_code=400)
    if "signal" in lowered or "signal_weekly" in lowered:
        return ApiErrorException("SIGNAL_INCOMPLETE", msg, status_code=400)
    if "weights" in lowered or "weight" in lowered:
        return ApiErrorException("WEIGHTS_SANITY_FAIL", msg, status_code=400)
    if "coverage" in lowered or "missing feature" in lowered:
        return ApiErrorException("INPUT_COVERAGE_MISSING", msg, status_code=400)
    return ApiErrorException("BAD_REQUEST", msg, status_code=400)
