from fastapi import Request
from fastapi.responses import JSONResponse
from app.exceptions.base import ProctoringException
from app.config import logger

async def proctoring_exception_handler(request: Request, exc: ProctoringException):
    logger.error(f"{request.url.path} — {exc.status_code} — {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message}
    )
