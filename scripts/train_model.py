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
from nslr import ood

SEED = 42


def synth_ood(X, y, n=300):
    """Fabricate out-of-distribution clips from real ones to test the reject gate:
    half are time-shuffled (motion order destroyed), half are cross-class mixes
    (first half of one sign + second half of another)."""
    rng = np.random.default_rng(SEED)
    half_t = X.shape[1] // 2
    out = []
    for _ in range(n // 2):
        c = X[rng.integers(len(X))].copy()
        rng.shuffle(c)                      # shuffle along the time axis
        out.append(c)
    for _ in range(n - n // 2):
        i = rng.integers(len(X))
        j = rng.integers(len(X))
        while y[j] == y[i]:
            j = rng.integers(len(X))
        out.append(np.concatenate([X[i][:half_t], X[j][half_t:]], axis=0))
    return np.stack(out).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Train + save one BiLSTM on all data.")
    p.add_argument("--processed", default=C.PROCESSED_DIR)
    p.add_argument("--results", default=C.RESULTS_DIR)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--threshold", type=float, default=0.75,
                   help="softmax confidence threshold stored for the live demo (proposal uses 0.85)")
    p.add_argument("--ood-percentile", type=float, default=99.0,
                   help="reject inputs farther than this percentile of in-distribution distances")
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
    present = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=present, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(present, weights)}

    model = build_bilstm(X.shape[1], X.shape[2], n_classes)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=a.epochs,
                     batch_size=a.batch_size, class_weight=class_weight, callbacks=[es],
                     verbose=a.verbose)

    os.makedirs(a.results, exist_ok=True)
    model_path = os.path.join(a.results, "model.keras")
    model.save(model_path)

    # --- fit the open-set rejection gate in the embedding space ---
    from sklearn.metrics import roc_auc_score
    emb = ood.embedding_model(model)
    Z = emb.predict(X, verbose=0)
    means, precision = ood.fit_prototypes(Z, y, n_classes)
    in_dist, _ = ood.mahalanobis_min(Z, means, precision)
    ood_threshold = float(np.percentile(in_dist, a.ood_percentile))

    Z_ood = emb.predict(synth_ood(X, y), verbose=0)
    ood_dist, _ = ood.mahalanobis_min(Z_ood, means, precision)
    auroc = roc_auc_score(np.r_[np.zeros(len(in_dist)), np.ones(len(ood_dist))],
                          np.r_[in_dist, ood_dist])
    rejected = float((ood_dist > ood_threshold).mean())
    false_reject = float((in_dist > ood_threshold).mean())
    ood.save_stats(os.path.join(a.results, "ood_stats.npz"), means, precision,
                   ood_threshold, a.ood_percentile)

    with open(os.path.join(a.results, "model_meta.json"), "w") as fh:
        json.dump({"class_names": class_names, "seq_len": int(X.shape[1]),
                   "confidence_threshold": a.threshold,
                   "ood_threshold": ood_threshold, "ood_percentile": a.ood_percentile,
                   "ood_stats": "ood_stats.npz"}, fh, indent=2)

    best_val = max(hist.history["val_accuracy"])
    print(f"\nBest val accuracy: {best_val:.3f} (over {len(hist.history['val_accuracy'])} epochs)")
    print(f"\nOpen-set gate (Mahalanobis): threshold={ood_threshold:.2f} @ p{a.ood_percentile:g}")
    print(f"  in-distribution distances: p50={np.percentile(in_dist,50):.2f} "
          f"p95={np.percentile(in_dist,95):.2f} max={in_dist.max():.2f}")
    print(f"  synthetic-OOD rejected: {rejected:.0%} | real signs wrongly rejected: {false_reject:.0%} "
          f"| separation AUROC={auroc:.3f}")
    print(f"Saved {model_path} + model_meta.json + ood_stats.npz")


if __name__ == "__main__":
    main()
