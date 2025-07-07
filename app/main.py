from fastapi import FastAPI
from contextlib import asynccontextmanager
from .config import logger
from .apis.identity_verification import router as id_router
from .exceptions.handler import proctoring_exception_handler
from .exceptions.base import ProctoringException
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup")
    yield
    logger.info("Application shutdown")

app = FastAPI(title="Video Proctoring Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:5500"] if you want to restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Identity Verification APIs
app.include_router(id_router)

# Custom Exception Handler
app.add_exception_handler(ProctoringException, proctoring_exception_handler)

# Health Check Endpoint
@app.get("/healthz", tags=["Health"])
def health_check():
    return {"status": "ok"}
