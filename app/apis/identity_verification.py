from fastapi import APIRouter, UploadFile, File
from app.exceptions.base import ProctoringException
from app.services.liveface_matching import (
    detect_faces, 
    match_with_databaseimages, 
    match_with_idproof,
    convert_pdf_to_face_image_bytes,
    preprocess_image_bytes
)
from app.exceptions.base import FaceNotFoundError
from app.config import logger
import imghdr



# Only this router declaration
router = APIRouter(prefix="/identity", tags=["Identity Verification"])

@router.post("/detect-face")
async def detect_face_from_live_frame(image: UploadFile = File(...)):
    data = await image.read()
    count, locations, image_base64 = detect_faces(data)
    if count == 0:
        raise FaceNotFoundError()
    logger.info(f"Detected {count} faces")
    return {
        "faces_detected": count,
        "locations": locations,
        "image_with_box": image_base64
    }

@router.post("/verify-with-databaseimages")
async def verify_with_database(live_image: UploadFile = File(...)):
    data = await live_image.read()
    preprocessed_data = preprocess_image_bytes(data)
    result = match_with_databaseimages(preprocessed_data)

    logger.info(f"DB match result: {result}")
    return result

@router.post("/verify-with-idproof")
async def verify_with_id(
    live_image: UploadFile = File(...),
    idproof_image: UploadFile = File(...)
):
    live = await live_image.read()
    idb = await idproof_image.read()    
    if not live or not idb:
        raise ProctoringException("Live image or ID proof image is empty", 400)

    try:
        filename = idproof_image.filename.lower()
        logger.info(f"Processing ID proof file: {filename}")
        preprocess_live = preprocess_image_bytes(live)

        if filename.endswith(".pdf"):
            logger.info("ID proof is a PDF. Converting to image...")
            preprocess_idb = convert_pdf_to_face_image_bytes(idb)
            logger.info("PDF successfully converted to image bytes")

        elif imghdr.what(None, idb) in ["jpeg", "png", "jpg"]:
            logger.info("ID proof is a valid image format")
            idb = preprocess_image_bytes(idb)

        else:
            raise ProctoringException("Unsupported file type. Please upload a valid image or PDF.", 400)

        logger.info("Calling match_with_idproof service function")
        result = match_with_idproof(preprocess_live, preprocess_idb)
        logger.info(f"ID proof match result: {result}")
        return result

    except ProctoringException:
        raise
    except Exception as e:
        logger.exception("ID proof processing failed")
        raise ProctoringException("Failed to verify ID proof", 500)