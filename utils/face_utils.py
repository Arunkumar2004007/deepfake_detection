"""
utils/face_utils.py — OpenCV + MTCNN face detection, alignment, landmark utility
"""
import cv2
import numpy as np
from config import Config

try:
    from mtcnn import MTCNN
    _detector = MTCNN()
    MTCNN_AVAILABLE = True
except Exception:
    MTCNN_AVAILABLE = False

# Haar cascade fallback
_HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_haar = cv2.CascadeClassifier(_HAAR_PATH)

TARGET_SIZE = Config.FACE_IMG_SIZE  # (224, 224)


def detect_face_mtcnn(frame_bgr: np.ndarray):
    """Detect face using MTCNN. Returns (x,y,w,h) or None."""
    if not MTCNN_AVAILABLE:
        return detect_face_haar(frame_bgr)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = _detector.detect_faces(rgb)
    if not results:
        return None
    best = max(results, key=lambda r: r["confidence"])
    x, y, w, h = best["box"]
    return max(x, 0), max(y, 0), w, h


def detect_face_haar(frame_bgr: np.ndarray):
    """Haar-cascade face detection fallback. Returns (x,y,w,h) or None."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return int(x), int(y), int(w), int(h)


def extract_and_align_face(frame_bgr: np.ndarray) -> np.ndarray | None:
    """
    Detect face, crop + align, resize to TARGET_SIZE.
    Returns float32 array [0,1] or None if no face found.
    """
    box = detect_face_mtcnn(frame_bgr)
    if box is None:
        return None
    x, y, w, h = box
    # Add a small margin
    margin = int(0.1 * max(w, h))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame_bgr.shape[1], x + w + margin)
    y2 = min(frame_bgr.shape[0], y + h + margin)
    face = frame_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face_resized = cv2.resize(face, TARGET_SIZE)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    return face_rgb.astype(np.float32) / 255.0


def preprocess_frame_for_model(frame_bgr: np.ndarray) -> np.ndarray | None:
    """
    Full preprocessing pipeline: detect → align → normalize.
    Returns shape (1, 224, 224, 3) ready for model or None.
    """
    face = extract_and_align_face(frame_bgr)
    if face is None:
        return None
    return np.expand_dims(face, axis=0)


def decode_base64_frame(b64_data: str) -> np.ndarray | None:
    """Decode a base64 image string (from canvas.toDataURL) to BGR numpy array."""
    import base64
    try:
        if "," in b64_data:
            b64_data = b64_data.split(",")[1]
        img_bytes = base64.b64decode(b64_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def get_face_landmarks(frame_bgr: np.ndarray) -> dict | None:
    """Return MTCNN keypoints (eyes, nose, mouth corners) or None."""
    if not MTCNN_AVAILABLE:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = _detector.detect_faces(rgb)
    if not results:
        return None
    return results[0].get("keypoints")


def eye_blink_score(keypoints: dict | None) -> float:
    """Approximate eye-openness ratio from keypoint geometry (0=closed, 1=open)."""
    if not keypoints:
        return 0.5
    try:
        le = np.array(keypoints["left_eye"])
        re = np.array(keypoints["right_eye"])
        ln = np.array(keypoints["left_mouth"])
        rn = np.array(keypoints["right_mouth"])
        eye_dist = np.linalg.norm(le - re)
        mouth_dist = np.linalg.norm(ln - rn)
        return float(np.clip(eye_dist / (mouth_dist + 1e-6), 0, 1))
    except Exception:
        return 0.5
