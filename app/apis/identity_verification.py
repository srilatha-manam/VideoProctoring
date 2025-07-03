from fastapi import APIRouter, UploadFile, File
from app.services.face_verification import (
    detect_faces, 
    match_with_databaseimages, 
    match_with_idproof
)
from app.exceptions.base import FaceNotFoundError
from app.config import logger

router = APIRouter(prefix="/identity", tags=["Identity Verification"])

@router.post("/detect-face")
async def detect_face_from_live_frame(image: UploadFile = File(...)):
    data = await image.read()
    count, locations = detect_faces(data)
    if count == 0:
        raise FaceNotFoundError()
    logger.info(f"Detected {count} faces")
    return {"faces_detected": count, "locations": locations}

@router.post("/verify-with-databaseimages")
async def verify_with_database(live_image: UploadFile = File(...)):
    data = await live_image.read()
    result = match_with_databaseimages(data)
    logger.info(f"DB match result: {result}")
    return result

@router.post("/verify-with-idproof")
async def verify_with_id(
    live_image: UploadFile = File(...),
    idproof_image: UploadFile = File(...)
):
    live = await live_image.read()
    idb  = await idproof_image.read()
    result = match_with_idproof(live, idb)
    logger.info(f"ID proof match result: {result}")
    return result
