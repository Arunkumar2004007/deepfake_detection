"""
models/similarity.py — Face and voice embedding similarity between baseline and live.

Methods:
  - Face: DeepFace ArcFace embeddings → cosine similarity
  - Voice: MFCC features → cosine similarity (DTW fallback)
"""
import numpy as np
from config import Config

THRESHOLD = Config.SIMILARITY_THRESHOLD  # 0.6


# ─── FACE SIMILARITY ─────────────────────────────────────────────────────────

def face_similarity(img_path1: str, img_path2: str) -> float:
    """
    Compare two face images using DeepFace ArcFace embeddings.
    Returns cosine similarity in [0, 1], 1 = identical.
    """
    try:
        from deepface import DeepFace
        result = DeepFace.verify(
            img1_path=img_path1,
            img2_path=img_path2,
            model_name="ArcFace",
            detector_backend="mtcnn",
            enforce_detection=False,
            distance_metric="cosine"
        )
        distance = float(result.get("distance", 1.0))
        # Convert cosine distance [0, 2] → similarity [0, 1]
        similarity = 1.0 - (distance / 2.0)
        return float(np.clip(similarity, 0.0, 1.0))
    except Exception as e:
        print(f"[Similarity] Face similarity error: {e}")
        return 0.5  # Neutral fallback


def face_embedding(img_path: str) -> np.ndarray | None:
    """Extract ArcFace embedding vector from an image file."""
    try:
        from deepface import DeepFace
        result = DeepFace.represent(
            img_path=img_path,
            model_name="ArcFace",
            detector_backend="mtcnn",
            enforce_detection=False
        )
        return np.array(result[0]["embedding"])
    except Exception as e:
        print(f"[Similarity] Embedding error: {e}")
        return None


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10
    return float(np.dot(v1, v2) / norm)


# ─── VOICE SIMILARITY ─────────────────────────────────────────────────────────

def voice_similarity(audio_path1: str, audio_path2: str) -> float:
    """
    Compare two audio files using MFCC cosine similarity.
    Returns similarity in [0, 1], 1 = identical voice.
    """
    from utils.audio_utils import compute_mfcc_similarity
    return compute_mfcc_similarity(audio_path1, audio_path2)


# ─── COMBINED SIMILARITY ──────────────────────────────────────────────────────

def combined_similarity(
    baseline_video: str, live_frame_path: str,
    baseline_audio: str, live_audio_path: str,
    face_weight: float = 0.6, voice_weight: float = 0.4
) -> dict:
    """
    Compute face + voice similarity between baseline and live recordings.

    Returns dict with face_sim, voice_sim, combined_sim, is_same_person.
    """
    face_sim  = face_similarity(baseline_video, live_frame_path)
    voice_sim = voice_similarity(baseline_audio, live_audio_path)

    total_w = face_weight + voice_weight
    combined = (face_weight / total_w) * face_sim + (voice_weight / total_w) * voice_sim
    combined = float(np.clip(combined, 0.0, 1.0))

    return {
        "face_similarity":     round(face_sim,  4),
        "voice_similarity":    round(voice_sim, 4),
        "combined_similarity": round(combined,  4),
        "is_same_person":      combined >= THRESHOLD,
        "threshold":           THRESHOLD
    }
