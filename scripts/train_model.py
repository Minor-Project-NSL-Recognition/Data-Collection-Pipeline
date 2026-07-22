"""Train ONE BiLSTM on ALL processed data and save it for live inference.

Unlike train_eval.py (which measures accuracy via leave-one-signer-out and keeps
no model), this trains a single deployable model on the full dataset — the best
model for the live demo, NOT a number to report.

    python -u scripts/train_model.py --verbose 2

Outputs to <results>/:
    model.keras        trained Keras model
    model_meta.json    class_names, seq_len, confidence_threshold
"""

import argparse
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nslr import config as C
from nslr.dataset import load_processed
from nslr.model import build_bilstm

SEED = 42


def main():
    p = argparse.ArgumentParser(description="Train + save one BiLSTM on all data.")
    p.add_argument("--processed", default=C.PROCESSED_DIR)
    p.add_argument("--results", default=C.RESULTS_DIR)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--threshold", type=float, default=0.75,
                   help="confidence threshold stored for the live demo (proposal uses 0.85)")
    p.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])
    a = p.parse_args()

    import random
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X, y, _, class_names = load_processed(a.processed)
    n_classes = len(class_names)
    print(f"X {X.shape} | classes {class_names}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=a.val_split, stratify=y, random_state=SEED)
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_tr)
    class_weight = {i: float(w) for i, w in enumerate(weights)}

    model = build_bilstm(X.shape[1], X.shape[2], n_classes)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=a.epochs,
                     batch_size=a.batch_size, class_weight=class_weight, callbacks=[es],
                     verbose=a.verbose)

    os.makedirs(a.results, exist_ok=True)
    model_path = os.path.join(a.results, "model.keras")
    model.save(model_path)
    with open(os.path.join(a.results, "model_meta.json"), "w") as fh:
        json.dump({"class_names": class_names, "seq_len": int(X.shape[1]),
                   "confidence_threshold": a.threshold}, fh, indent=2)

    best_val = max(hist.history["val_accuracy"])
    print(f"\nBest val accuracy: {best_val:.3f} (over {len(hist.history['val_accuracy'])} epochs)")
    print(f"Saved {model_path} + model_meta.json")


if __name__ == "__main__":
    main()
