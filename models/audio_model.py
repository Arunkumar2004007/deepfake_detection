"""
models/audio_model.py — CNN + Bidirectional LSTM Audio Deepfake Detector

Architecture:
    Input:  (40, 128, 1)   — MFCC spectrogram
    CNN:    Conv2D(32) → BN → MaxPool → Conv2D(64) → BN → MaxPool
            → Conv2D(128) → BN → MaxPool
    Reshape: into sequence of frames for LSTM
    LSTM:   Bidirectional LSTM(128) → Dropout(0.4)
            → Bidirectional LSTM(64)  → Dropout(0.3)
    Head:   Dense(128, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
    Output: scalar in [0, 1]  (0=Real, 1=Fake)
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from config import Config
from utils.audio_utils import preprocess_audio_for_model, preprocess_audio_file, compute_mfcc_similarity

N_MFCC      = Config.N_MFCC      # 40
MAX_PAD_LEN = Config.MAX_PAD_LEN  # 128
MODEL_PATH  = Config.AUDIO_MODEL_PATH

_model: keras.Model | None = None


# ─── MODEL DEFINITION ─────────────────────────────────────────────────────────

def build_audio_model() -> keras.Model:
    """Build CNN + BiLSTM audio deepfake detection model."""
    inputs = keras.Input(shape=(N_MFCC, MAX_PAD_LEN, 1), name="mfcc_input")

    # CNN block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # CNN block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # CNN block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Reshape for LSTM: (batch, time_steps, features)
    shape = x.shape
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)

    # BiLSTM blocks
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)

    # Classification head
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid", name="fake_score")(x)

    model = Model(inputs, output, name="AudioDeepfakeDetector")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model


# ─── LOAD / CACHE ─────────────────────────────────────────────────────────────

def get_audio_model() -> keras.Model:
    """Return cached model, loading from disk if available."""
    global _model
    if _model is not None:
        return _model
    if os.path.exists(MODEL_PATH):
        try:
            _model = keras.models.load_model(MODEL_PATH)
            print(f"[AudioModel] Loaded from {MODEL_PATH}")
            return _model
        except Exception as e:
            print(f"[AudioModel] Failed to load: {e}. Building fresh model.")
    _model = build_audio_model()
    print("[AudioModel] Using randomly initialized weights (train for real accuracy).")
    return _model


# ─── INFERENCE ────────────────────────────────────────────────────────────────

def predict_audio_file(path: str) -> dict:
    """
    Predict fake probability for an audio file.
    Returns dict: score, label, confidence.
    """
    model = get_audio_model()
    try:
        mfcc_input = preprocess_audio_for_model(path)
        score = float(model.predict(mfcc_input, verbose=0)[0][0])
    except Exception as e:
        print(f"[AudioModel] Prediction error: {e}")
        score = 0.0
    label = "fake" if score >= Config.AUDIO_THRESHOLD else "real"
    return {
        "score": round(score, 4),
        "label": label,
        "confidence": round(abs(score - 0.5) * 2, 4)
    }


def predict_audio_bytes(audio_bytes: bytes, sample_rate: int = 16000) -> dict:
    """
    Predict from raw PCM audio bytes (from WebRTC stream).
    """
    import tempfile, soundfile as sf, numpy as np
    try:
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio_array /= 32768.0
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_array, sample_rate)
            return predict_audio_file(tmp.name)
    except Exception as e:
        print(f"[AudioModel] predict_audio_bytes error: {e}")
        return {"score": 0.0, "label": "real", "confidence": 0.0}
