from fastapi import APIRouter, UploadFile, File
from app.exceptions.base import ProctoringException
from app.services.liveface_matching import (
    detect_faces, 
    match_with_databaseimages, 
    match_with_idproof
)
from app.exceptions.base import FaceNotFoundError
from app.config import logger
import cv2
import numpy as np
import imghdr
from pdf2image import convert_from_bytes

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
    result = match_with_databaseimages(data)
    logger.info(f"DB match result: {result}")
    return result

@router.post("/verify-with-idproof")
async def verify_with_id(
    live_image: UploadFile = File(...),
    idproof_image: UploadFile = File(...)
):
    # Read the image files
    live = await live_image.read()
    idb = await idproof_image.read()    
    print("capture live image and id proof image")
    if not live or not idb:
        raise ProctoringException("Live image or ID proof image is empty", 400)

    try:
        filename = idproof_image.filename.lower()
        logger.info(f"Processing ID proof file: {filename}")
        print(f"Processing ID proof file: {filename}")
        # Handle PDF ID proof
        if filename.endswith(".pdf"):
            images = convert_from_bytes(idb, dpi=300)
            if not images:
                raise ProctoringException("No image found in PDF", 400)

            # Convert PIL to OpenCV image
            pil_image = images[0].convert("RGB")
            open_cv_image = np.array(pil_image)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

            # Encode as JPEG bytes
            success, encoded_img = cv2.imencode('.jpg', open_cv_image)
            if not success:
                raise ProctoringException("Failed to encode image from PDF", 500)
            else:
                print("Successfully encoded image from PDF")
            idb = encoded_img.tobytes()
            logger.info("Converted PDF to image bytes successfully")            

        # Handle regular image formats
        elif imghdr.what(None, idb) in ["jpeg", "png", "jpg"]:
            logger.info("ID proof is a valid image format")

        else:
            raise ProctoringException("Unsupported file type. Please upload a valid image or PDF.", 400)

        # Call the face matching and OCR service
        logger.info("Calling match_with_idproof service function")       
        result = match_with_idproof(live, idb)
        logger.info(f"ID proof match result: {result}")
        return result

    except ProctoringException:
        raise
    except Exception as e:
        logger.exception("ID proof processing failed")
        raise ProctoringException("Failed to verify ID proof", 500)