from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.config import logger
from app.api.identity_verification import router as id_router
from app.exceptions.handlers import proctoring_exception_handler
from app.exceptions.base import ProctoringException

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup")
    yield
    logger.info("Application shutdown")

app = FastAPI(title="Video Proctoring Service", lifespan=lifespan)

# Identity Verification APIs
app.include_router(id_router)

# Custom Exception Handler
app.add_exception_handler(ProctoringException, proctoring_exception_handler)

# Health Check Endpoint
@app.get("/healthz", tags=["Health"])
def health_check():
    return {"status": "ok"}
