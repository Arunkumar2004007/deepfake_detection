
"""
train_audio_model.py — Training pipeline for CNN+BiLSTM audio deepfake detector

Dataset: ASVspoof 2019 / 2021
  Folder structure:
    data_audio/
      real/   → genuine speech .wav files
      fake/   → spoofed speech .wav files

Usage:
  python train_audio_model.py --data_dir data_audio/ --epochs 30
"""
import os, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score)

from models.audio_model import build_audio_model
from utils.audio_utils   import preprocess_audio_file
from config import Config

MODEL_PATH = Config.AUDIO_MODEL_PATH


def load_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for label, folder in [(0, "real"), (1, "fake")]:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            print(f"[Train] Warning: {folder_path} not found")
            continue
        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith((".wav", ".flac", ".mp3", ".ogg"))]
        for fname in files:
            path = os.path.join(folder_path, fname)
            try:
                mfcc = preprocess_audio_file(path)  # (40, 128)
                X.append(mfcc[..., np.newaxis])     # (40, 128, 1)
                y.append(float(label))
            except Exception as e:
                print(f"[Train] Skipping {fname}: {e}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train(args):
    print(f"[Train] Loading audio dataset from {args.data_dir}")
    X, y = load_dataset(args.data_dir)

    if len(X) == 0:
        print("[Train] No audio files found. Place .wav files in data_audio/real/ and data_audio/fake/")
        return

    print(f"[Train] Loaded {len(X)} files | Real: {int((y==0).sum())} | Fake: {int((y==1).sum())}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(len(X_train)).batch(32).prefetch(tf.data.AUTOTUNE))
    val_ds   = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
                .batch(32).prefetch(tf.data.AUTOTUNE))

    model = build_audio_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_auc", mode="max"),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.3, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_auc", mode="max"),
    ]

    print(f"[Train] Training for up to {args.epochs} epochs...")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # Evaluation
    print("\n[Train] Evaluating on test set...")
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred      = (y_pred_prob >= 0.5).astype(int)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_prob):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\n[Train] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Audio Deepfake Detector")
    parser.add_argument("--data_dir", default="data_audio", help="Path to audio dataset folder")
    parser.add_argument("--epochs",   default=30, type=int)
    args = parser.parse_args()
    train(args)
