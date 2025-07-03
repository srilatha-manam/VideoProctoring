import os
import tempfile
import numpy as np
import face_recognition
from deepface import DeepFace
from app.exceptions.base import ProctoringException
from app.config import logger

DATABASE_FOLDER = "face_db"

def detect_faces(image_bytes: bytes):
    # Load the image from the bytes
    img = face_recognition.load_image_file(image_bytes)
    # Detect the faces in the image
    locs = face_recognition.face_locations(img)
    # Return the number of faces and their locations
    return len(locs), locs

def match_with_databaseimages(live_bytes: bytes):
    # Detect faces in the live image
    faces, _ = detect_faces(live_bytes)
    # If no faces are detected, raise an exception
    if faces == 0:
        raise ProctoringException("No face in live image", 422)

    # Encode the face in the live image
    live_encoding = face_recognition.face_encodings(face_recognition.load_image_file(live_bytes))[0]

    # Iterate through the database images
    for fn in os.listdir(DATABASE_FOLDER):  #change with direct image path curently
        # Check if the file is a jpg or png image
        if fn.lower().endswith((".jpg", ".png")):
            # Get the path of the image
            path = os.path.join(DATABASE_FOLDER, fn)
            # Encode the face in the database image
            encodings = face_recognition.face_encodings(face_recognition.load_image_file(path))
            # If a face is detected in the database image and it matches the live image, return the filename
            if encodings and face_recognition.compare_faces([encodings[0]], live_encoding)[0]:
                return {"success": True, "matched_with": fn}
    # If no match is found, return a message
    return {"success": False, "message": "Your face does not match with any database image."}

def match_with_idproof(live_bytes: bytes, id_bytes: bytes):
    # DeepFace will handle blur if enforce_detection=False
    try:
        # Create temporary files to store the live and id proof images
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as a, \
             tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as b:  #check is this jpg pr png
            # Write the live and id proof images to the temporary files
            a.write(live_bytes);  a.flush()
            b.write(id_bytes);    b.flush()
            # Use DeepFace to verify the match between the live and id proof images
            result = DeepFace.verify(a.name, b.name, enforce_detection=False)
            return {
                "success": result["verified"],
                "distance": result["distance"],
                "message": "Match" if result["verified"] else "No match"
            }
    except Exception as e:
        logger.exception("DeepFace verification failure")
        raise ProctoringException("ID proof verification error", 500)
