class ProctoringException(Exception):
    """Base exception for all proctoring errors."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

class FaceNotFoundError(ProctoringException):
    """Raised when no face is detected in the input image."""
    def __init__(self):
        super().__init__("No face found in the provided image", 422)
