import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from app.exceptions.base import ProctoringException
from app.config import logger
import base64
import easyocr
from pdf2image import convert_from_bytes

# Load face analysis once globally
face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)
ocr_reader = easyocr.Reader(['en'])

TEST_IMAGE_PATH = r"C:\Users\Srilatha\Downloads\test_photo.jpg"  # for test image match


def detect_faces(image_bytes: bytes):
    try:
        # Check if image data is empty or corrupted
        if not image_bytes:
            raise ProctoringException("Image data is empty or corrupted", 400)

        # Convert bytes to image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect faces
        faces = face_app.get(img)
        bboxes = [f.bbox.astype(int).tolist() for f in faces]

        # Draw bounding boxes on the image
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert image with boxes to base64
        _, buffer = cv2.imencode('.jpg', img)
        img_bytes = buffer.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        return len(faces), bboxes, img_base64

    except Exception as e:
        logger.exception("Face detection failed")
        raise ProctoringException("Face detection error", 500)


def extract_embedding(image_bytes: bytes):
    # Convert the image bytes to a numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode the numpy array into an image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)    
    # Use the face application to detect faces in the imageS
    faces = face_app.get(img)    
    if not faces:
        raise ProctoringException("No face found in image", 422)
    if len(faces) > 1:
        raise ProctoringException("Multiple faces detected. Only one allowed.", 422)
    return faces[0].embedding


def match_with_databaseimages(live_bytes: bytes):  #need to modify this once the data base ready
    # Try to match the live bytes with the test image
    try:
        # Extract the embedding from the live bytes
        live_embedding = extract_embedding(live_bytes)

        # Check if the test image exists
        if not os.path.exists(TEST_IMAGE_PATH):
            # Raise an exception if the test image does not exist
            raise ProctoringException("Test image not found at given path", 500)

        # Open the test image
        with open(TEST_IMAGE_PATH, "rb") as f:   #change this with data base code once ready
            # Read the bytes from the test image
            test_bytes = f.read()

        # Extract the embedding from the test image
        test_embedding = extract_embedding(test_bytes)

        # Cosine similarity
        sim = np.dot(live_embedding, test_embedding) / (
            np.linalg.norm(live_embedding) * np.linalg.norm(test_embedding)
        )

        matched = bool(sim >= 0.3)  # Convert numpy.bool_ to native Python bool
  # you can tune threshold

        return {
            "success": matched,
            "matched_with": os.path.basename(TEST_IMAGE_PATH) if matched else None,
            "similarity_score": round(float(sim), 4),
            "message": "Match" if matched else "No match",
        }

    except ProctoringException:
        raise
    except Exception as e:
        logger.exception("Error matching with test image")
        raise ProctoringException("Face match error", 500)


def match_with_aadhar(live_bytes: bytes, id_bytes: bytes):
    # Try to extract embeddings from live and id bytes
    try:        
        logger.info("Extracting embedding from live image")
        live_embedding = extract_largest_face_embedding(live_bytes)        
        logger.info("Extracting embedding from ID proof")
        id_embedding = extract_largest_face_embedding(id_bytes)
        # Calculate the similarity score between the two embeddings
        sim = np.dot(live_embedding, id_embedding) / (
            np.linalg.norm(live_embedding) * np.linalg.norm(id_embedding)
        )
        matched = bool(sim >= 0.3)  # Adjustable threshold        
        # Extract text from the ID proof
        extracted_text = extract_text_from_id(id_bytes)      

        # Return a dictionary containing the success status, similarity score, message, and extracted text
        return {
            "success": matched,
            "similarity_score": round(float(sim), 4),
            "message": "Match" if matched else "No match",
            "id_text_summary": extracted_text
        }

    except ProctoringException:
        raise
    except Exception as e:
        # Log the exception and raise a ProctoringException with a message and status code
        logger.exception("ID proof verification failed")
        raise ProctoringException("Internal server error during ID verification", 500)
# def extract_largest_face_embedding(image_bytes: bytes):
#     try:        
#         nparr = np.frombuffer(image_bytes, np.uint8)        
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)       
#         if img is None:
#             raise ProctoringException("Failed to decode image from bytes", 422)        
#         faces = face_app.get(img)        
#         if not faces:
#             raise ProctoringException("No face found in image", 422)

#         # Sort by bounding box area
#         faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
#         return faces[0].embedding
#     except Exception as e:
#         logger.exception("Error in extracting face embedding")
#         raise
def extract_largest_face_embedding(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ProctoringException("Failed to decode image", 422)

    faces = face_app.get(img)
    if not faces:
        raise ProctoringException("No face found in image", 422)

    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].embedding

def extract_text_from_id(image_bytes: bytes):
    # Convert the image bytes to a numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode the numpy array into an image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Use the OCR reader to read the text from the image
    results = ocr_reader.readtext(img)   
    # Join the extracted text into a single string
    extracted_text = " ".join([text[1] for text in results])
    # Return the extracted text
    return extracted_text

def convert_pdf_to_face_image_bytes(pdf_bytes: bytes) -> bytes:
    # Convert PDF bytes to images
    images = convert_from_bytes(pdf_bytes, dpi=300)
    if not images:
        raise ProctoringException("No image found in PDF", 400)

    # Convert first page to OpenCV
    pil_image = images[0].convert("RGB")
    open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Optional: Rotate if landscape
    if open_cv_image.shape[1] > open_cv_image.shape[0]:
        open_cv_image = cv2.rotate(open_cv_image, cv2.ROTATE_90_CLOCKWISE)

    # Resize large images
    max_dim = 1000
    h, w = open_cv_image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        open_cv_image = cv2.resize(open_cv_image, None, fx=scale, fy=scale)

    # Enhance contrast
    open_cv_image = cv2.convertScaleAbs(open_cv_image, alpha=1.3, beta=30)
    # debug face detection before converting
    faces = face_app.get(open_cv_image)
    logger.info(f"[convert_pdf_to_face_image_bytes] Faces detected in PDF image: {len(faces)}")
    if not faces:
        raise ProctoringException("No face detected in ID proof (converted from PDF)", 422)

    # Encode to JPEG bytes
    success, encoded_img = cv2.imencode(".jpg", open_cv_image)
    if not success:
        raise ProctoringException("Failed to encode image from PDF", 500)

    return encoded_img.tobytes()

def preprocess_image_bytes(image_bytes: bytes) -> bytes:
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode the image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Raise an exception if the image fails to decode
    if img is None:
        raise ProctoringException("Failed to decode image", 422)

    # Rotate if in landscape
    if img.shape[1] > img.shape[0]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # Resize large images
    max_dim = 1000
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, None, fx=scale, fy=scale)

    # Enhance contrast
    img = cv2.convertScaleAbs(img, alpha=1.3, beta=30)

    # Encode back to bytes
    success, encoded_img = cv2.imencode(".jpg", img)
    if not success:
        raise ProctoringException("Failed to encode preprocessed image", 500)
    return encoded_img.tobytes()
def preprocess_pan_image(image_bytes: bytes) -> bytes:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ProctoringException("Failed to decode PAN card image", 422)

    # Convert to grayscale (PAN is usually grayscale or low-contrast)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to enhance contrast
    gray = cv2.equalizeHist(gray)

    # Resize if too small
    if min(gray.shape[:2]) < 300:
        gray = cv2.resize(gray, (600, 400))

    # Convert back to BGR for InsightFace
    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    success, encoded_img = cv2.imencode(".jpg", img_bgr)
    if not success:
        raise ProctoringException("Failed to encode preprocessed PAN image", 500)
    return encoded_img.tobytes()

def extract_text_from_pan(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = ocr_reader.readtext(img)
    extracted_text = " ".join([text[1] for text in results])
    return extracted_text

def match_with_pan_card(live_bytes: bytes, pan_bytes: bytes):
    try:
        live_embedding = extract_largest_face_embedding(live_bytes)
        pan_embedding = extract_largest_face_embedding(pan_bytes)

        sim = np.dot(live_embedding, pan_embedding) / (
            np.linalg.norm(live_embedding) * np.linalg.norm(pan_embedding)
        )
        matched = bool(sim >= 0.3)
        ocr_text = extract_text_from_pan(pan_bytes)

        return {
            "success": matched,
            "similarity_score": round(float(sim), 4),
            "message": "Match" if matched else "No match",
            "pan_text_summary": ocr_text
        }
    except ProctoringException:
        raise
    except Exception as e:
        logger.exception("PAN card face matching failed")
        raise ProctoringException("PAN card matching failed", 500)