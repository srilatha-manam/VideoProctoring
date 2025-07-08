from fastapi import APIRouter, UploadFile, File
from app.services.background_audio import extract_audio
from app.exceptions.base import ProctoringException
from app.config import logger

router = APIRouter(prefix="/audio-proctoring", tags=["Audio Proctoring"])

@router.post("/background_conversations")
async def background_conversations_from_video(video_file: UploadFile = File(...)):
    if not video_file:
        raise ProctoringException("No video file provided", 400)

    try:
        video_bytes = await video_file.read()
        result = extract_audio(video_bytes, video_file.filename)
        return result

    except Exception as e:
        logger.exception("VAD processing failed")
        raise ProctoringException("Failed to process video for VAD", 500)
