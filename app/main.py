from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.config import logger
from app.apis.identity_verification import router as id_router
from app.apis.voice_activity_detection import router as aa_router
from app.exceptions.handler import proctoring_exception_handler
from app.exceptions.base import ProctoringException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup")
    yield
    logger.info("Application shutdown")

app = FastAPI(title="Video Proctoring Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Identity Verification APIs
app.include_router(id_router)
app.include_router(aa_router)

# Custom Exception Handler
app.add_exception_handler(ProctoringException, proctoring_exception_handler)

# Health Check Endpoint
@app.get("/healthz", tags=["Health"])
def health_check():
    return {"status": "ok"}
