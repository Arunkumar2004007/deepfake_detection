"""
models/video_model.py — CNN Video Deepfake Detector (EfficientNetB4 + custom head)

Architecture:
    Input:  (224, 224, 3)  — normalized RGB face crop
    Base:   EfficientNetB4 (pretrained ImageNet, top removed)
    Head:   GlobalAveragePooling → Dense(512, ReLU) → Dropout(0.4)
            → Dense(256, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
    Output: scalar in [0, 1]  (0=Real, 1=Deepfake)

v5 changes:
    - Rolling window buffer (last 8 frames) for temporal signal computation
    - Eye blink and landmark stability analysed across buffered frames
    - Signal keys updated to match 8-signal heuristic_detector v5
"""
import os
import collections
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4

from config import Config
from utils.face_utils import preprocess_frame_for_model, decode_base64_frame

IMG_SIZE = Config.FACE_IMG_SIZE  # (224, 224)
MODEL_PATH = Config.VIDEO_MODEL_PATH

_model: keras.Model | None = None

# ── Rolling frame buffer for temporal signals ─────────────────────────────────
# Keyed by session_id. Stores last N decoded BGR frames.
_BUFFER_SIZE = 8
_frame_buffers: dict = {}   # session_id → deque[np.ndarray]


def _get_buffer(session_id: str) -> collections.deque:
    if session_id not in _frame_buffers:
        _frame_buffers[session_id] = collections.deque(maxlen=_BUFFER_SIZE)
    return _frame_buffers[session_id]


def reset_buffer(session_id: str) -> None:
    """Clear the rolling frame buffer for a session (call on session start)."""
    _frame_buffers.pop(session_id, None)


# ─── MODEL DEFINITION ─────────────────────────────────────────────────────────

def build_video_model() -> keras.Model:
    """Build and return EfficientNetB4-based deepfake detector."""
    # Build base model with version-safe dropout rate argument.
    # 'drop_connect_rate' was removed in Keras 3 / TF 2.16+.
    _base_kwargs = dict(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    try:
        base = EfficientNetB4(**_base_kwargs, drop_connect_rate=0.4)
    except TypeError:
        try:
            base = EfficientNetB4(**_base_kwargs, survival_probability=0.6)  # Keras 3+
        except TypeError:
            base = EfficientNetB4(**_base_kwargs)  # plain fallback
    # Freeze base initially; unfreeze top 20 layers for fine-tuning
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="face_input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu", name="fc2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid", name="deepfake_score")(x)

    model = Model(inputs, output, name="VideoDeepfakeDetector")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model


# ─── LOAD / CACHE ─────────────────────────────────────────────────────────────

def get_video_model() -> keras.Model:
    """Return cached model, loading from disk if available."""
    global _model
    if _model is not None:
        return _model
    if os.path.exists(MODEL_PATH):
        try:
            _model = keras.models.load_model(MODEL_PATH)
            print(f"[VideoModel] Loaded from {MODEL_PATH}")
            return _model
        except Exception as e:
            print(f"[VideoModel] Failed to load: {e}. Building fresh model.")
    _model = build_video_model()
    print("[VideoModel] Using randomly initialized weights (train for real accuracy).")
    return _model


# ─── INFERENCE ────────────────────────────────────────────────────────────────

def predict_frame(frame_bgr: np.ndarray) -> float:
    """
    Predict deepfake probability for a single BGR frame.
    Returns float in [0, 1]: 0=real, 1=deepfake.
    """
    model = get_video_model()
    processed = preprocess_frame_for_model(frame_bgr)
    if processed is None:
        return 0.0   # no face detected → treat as real
    score = float(model.predict(processed, verbose=0)[0][0])
    return score


def predict_base64_frame(b64_data: str, session_id: str = "default") -> dict:
    """
    Decode a base64 frame and run deepfake prediction.

    Uses rolling window buffer (last 8 frames) to compute temporal signals
    (eye blink, landmark stability) even during live streaming.

    PRIMARY: Heuristic detector (11 signals, v6 GAN fingerprint).
    FALLBACK: CNN only when a trained model file exists on disk.

    Returns dict with score, label, confidence, liveness, and per-signal breakdown.
    """
    from utils.face_utils import decode_base64_frame
    from models.heuristic_detector import (
        score_frame_detailed, THRESHOLD as H_THRESH,
        _eye_blink_score, _landmark_stability_score, _temporal_flicker,
    )
    import cv2

    frame = decode_base64_frame(b64_data)
    if frame is None:
        return {"score": 0.0, "label": "no_face", "confidence": 0.0,
                "signals": {}, "liveness": {}}

    # ── Add frame to rolling buffer ───────────────────────────────────────────
    buf = _get_buffer(session_id)
    buf.append(frame)
    buf_frames = list(buf)

    # ── Heuristic: static frame signals (incl. GAN fingerprint) ──────────────
    detail  = score_frame_detailed(frame)
    h_score = detail["score"]
    gan_score = detail.get("gan", 0.0)

    # ── Temporal signals from buffer ──────────────────────────────────────────
    blink_score    = 0.3
    landmark_score = 0.3
    flicker_score  = 0.0
    if len(buf_frames) >= 3:
        try:
            blink_score    = _eye_blink_score(buf_frames)
        except Exception:
            pass
        try:
            landmark_score = _landmark_stability_score(buf_frames)
        except Exception:
            pass
        try:
            grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in buf_frames]
            flicker_score = _temporal_flicker(grays)
        except Exception:
            pass

    # ── Incorporate temporal into score ──────────────────────────────────────
    from models.heuristic_detector import _WT
    frame_w = 1.0 - (_WT["eye_blink"] + _WT["landmark"] + _WT["flicker"])
    temporal_adj = (
        _WT["eye_blink"] * blink_score +
        _WT["landmark"]  * landmark_score +
        _WT["flicker"]   * flicker_score
    )
    h_score_full = float(np.clip(frame_w * h_score + temporal_adj, 0.0, 1.0))

    # ── DUAL OVERRIDE RULES (v6) ─────────────────────────────────────────────
    # RULE 1 — Strong GAN fingerprint → floor raised to 0.55 (AI signal wins)
    if gan_score > 0.72:
        h_score_full = max(h_score_full, 0.55)
    # RULE 2 — Clear blink + low GAN → definitely real, cap at 0.45
    elif blink_score < 0.10 and gan_score < 0.35:
        h_score_full = min(h_score_full, 0.45)

    # ── CNN path (only if a trained model is saved) ───────────────────────────
    final_score = h_score_full
    if os.path.exists(MODEL_PATH):
        try:
            cnn_score   = predict_frame(frame)
            # Blend: 35% CNN + 65% heuristic
            final_score = 0.35 * cnn_score + 0.65 * h_score_full
        except Exception:
            pass

    label = "deepfake" if final_score >= H_THRESH else "real"

    # Update signal dict with temporal values
    signals = dict(detail)
    signals["eye_blink"]          = round(blink_score,    4)
    signals["landmark_stability"] = round(landmark_score, 4)
    signals["temporal_flicker"]   = round(flicker_score,  4)

    liveness = {
        "blink_detected"  : blink_score < 0.15,
        "eye_blink_score" : round(blink_score, 4),
        "natural_movement": landmark_score < 0.40,
        "gan_fingerprint" : round(gan_score, 4),
        "gan_detected"    : gan_score > 0.55,
    }

    return {
        "score"     : round(final_score, 4),
        "label"     : label,
        "confidence": round(abs(final_score - 0.5) * 2, 4),
        "signals"   : signals,
        "liveness"  : liveness,
    }




def predict_video_file(video_path: str, sample_every_n: int = 5,
                       max_frames: int = 20) -> float:
    """
    Process a video file by sampling up to `max_frames` frames evenly
    distributed across the whole video.  This keeps analysis fast
    regardless of video length.
    """
    import cv2
    cap        = cv2.VideoCapture(video_path)
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps        = cap.get(cv2.CAP_PROP_FPS) or 25

    if total <= 0:
        # Fallback: read sequentially, cap at max_frames
        scores    = []
        frame_idx = 0
        while len(scores) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % max(1, sample_every_n) == 0:
                scores.append(predict_frame(frame))
            frame_idx += 1
        cap.release()
        return float(np.mean(scores)) if scores else 0.0

    # Evenly distribute max_frames sample positions across the full video
    n_samples  = min(max_frames, total)
    step       = max(1, total // n_samples)
    positions  = [i * step for i in range(n_samples) if i * step < total]

    scores = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret and frame is not None:
            scores.append(predict_frame(frame))

    cap.release()
    return float(np.mean(scores)) if scores else 0.0
