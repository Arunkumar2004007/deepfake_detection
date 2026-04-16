"""
train_video_model.py — Training pipeline for CNN video deepfake detector

Dataset: FaceForensics++ or DFDC
  Folder structure expected:
    data/
      real/   → real face images/frames
      fake/   → deepfake face images/frames

Usage:
  python train_video_model.py --data_dir data/ --epochs 30 --batch_size 32
"""
import os, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score)

from models.video_model import build_video_model
from config import Config

IMG_SIZE    = Config.FACE_IMG_SIZE
MODEL_PATH  = Config.VIDEO_MODEL_PATH


def load_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load images from data_dir/{real, fake} folders."""
    import cv2
    X, y = [], []
    for label, folder in [(0, "real"), (1, "fake")]:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            print(f"[Train] Warning: {folder_path} not found")
            continue
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            img = cv2.imread(os.path.join(folder_path, fname))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            X.append(img.astype(np.float32) / 255.0)
            y.append(float(label))
    return np.array(X), np.array(y)


def augment_dataset(X: np.ndarray, y: np.ndarray):
    """Build tf.data augmentation pipeline."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    augment = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomBrightness(0.2),
        keras.layers.RandomContrast(0.2),
    ])
    dataset = (dataset
               .shuffle(buffer_size=len(X))
               .map(lambda x, l: (augment(x, training=True), l),
                    num_parallel_calls=tf.data.AUTOTUNE)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE))
    return dataset


def train(args):
    print(f"[Train] Loading dataset from {args.data_dir}")
    X, y = load_dataset(args.data_dir)

    if len(X) == 0:
        print("[Train] No images found. Place images in data/real/ and data/fake/")
        return

    print(f"[Train] Loaded {len(X)} images | Real: {int((y==0).sum())} | Fake: {int((y==1).sum())}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

    train_ds = augment_dataset(X_train, y_train)
    val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)

    model = build_video_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_auc", mode="max"),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_auc", mode="max"),
    ]

    print(f"[Train] Training for up to {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Evaluation on test set
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
    parser = argparse.ArgumentParser(description="Train Video Deepfake Detector")
    parser.add_argument("--data_dir", default="data",   help="Path to dataset folder")
    parser.add_argument("--epochs",   default=30, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    args = parser.parse_args()
    train(args)
