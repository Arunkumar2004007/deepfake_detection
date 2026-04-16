"""
utils/audio_utils.py — Librosa-based MFCC extraction and audio preprocessing
"""
import numpy as np
import librosa
import io
from config import Config

SAMPLE_RATE = Config.SAMPLE_RATE   # 16000
N_MFCC      = Config.N_MFCC       # 40
MAX_PAD_LEN = Config.MAX_PAD_LEN   # 128


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio file and resample to SAMPLE_RATE."""
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y, sr


def extract_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract MFCC features.
    Returns padded/truncated array of shape (N_MFCC, MAX_PAD_LEN).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    # Pad or truncate to fixed length
    if mfcc.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]
    return mfcc.astype(np.float32)


def extract_mel_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract log-mel spectrogram of shape (128, MAX_PAD_LEN)."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    if log_S.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - log_S.shape[1]
        log_S = np.pad(log_S, ((0, 0), (0, pad_width)), mode="constant")
    else:
        log_S = log_S[:, :MAX_PAD_LEN]
    return log_S.astype(np.float32)


def extract_pitch(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Estimate pitch (fundamental frequency) curve."""
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch_values.append(pitches[index, t])
    pitch = np.array(pitch_values, dtype=np.float32)
    if len(pitch) < MAX_PAD_LEN:
        pitch = np.pad(pitch, (0, MAX_PAD_LEN - len(pitch)))
    else:
        pitch = pitch[:MAX_PAD_LEN]
    return pitch


def reduce_noise(y: np.ndarray) -> np.ndarray:
    """Simple spectral noise reduction using noisereduce."""
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=y, sr=SAMPLE_RATE)
    except ImportError:
        return y


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """Normalize audio amplitude to [-1, 1]."""
    max_val = np.max(np.abs(y))
    if max_val > 0:
        return y / max_val
    return y


def preprocess_audio_file(path: str) -> np.ndarray:
    """
    Full pipeline: load → normalize → noise_reduce → extract MFCC.
    Returns shape (N_MFCC, MAX_PAD_LEN) = (40, 128).
    """
    y, sr = load_audio(path)
    y = normalize_audio(y)
    y = reduce_noise(y)
    return extract_mfcc(y, sr)


def preprocess_audio_for_model(path: str) -> np.ndarray:
    """
    Returns shape (1, N_MFCC, MAX_PAD_LEN, 1) ready for CNN+LSTM model.
    """
    mfcc = preprocess_audio_file(path)
    return np.expand_dims(np.expand_dims(mfcc, axis=0), axis=-1)


def compute_mfcc_similarity(path1: str, path2: str) -> float:
    """
    Compare two audio files using MFCC cosine similarity.
    Returns value in [0, 1] where 1 = identical.
    """
    try:
        mfcc1 = preprocess_audio_file(path1).flatten()
        mfcc2 = preprocess_audio_file(path2).flatten()
        dot   = np.dot(mfcc1, mfcc2)
        norm  = (np.linalg.norm(mfcc1) * np.linalg.norm(mfcc2)) + 1e-10
        return float(np.clip(dot / norm, 0.0, 1.0))
    except Exception:
        return 0.5
