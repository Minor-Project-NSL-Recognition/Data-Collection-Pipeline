"""Train + evaluate the BiLSTM pilot: does the seed data separate into 6 classes?

Two protocols: leave-one-signer-out (signer-independent, the headline metric) and
a stratified random-split baseline. Saves metrics.json + confusion matrix + curves.

    python -u scripts/train_eval.py                 # -u = live, unbuffered output
    python -u scripts/train_eval.py --verbose 2     # one line per epoch
    python scripts/train_eval.py --epochs 200 --batch-size 16
"""

import argparse
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nslr import config as C
from nslr.dataset import eligible_test_signers, load_processed
from nslr.model import build_bilstm

SEED = 42


def train_one(X_tr, y_tr, X_val, y_val, n_classes, epochs, batch_size, verbose):
    from tensorflow import keras
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.arange(n_classes)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    model = build_bilstm(X_tr.shape[1], X_tr.shape[2], n_classes)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, class_weight=class_weight,
                        callbacks=[es], verbose=verbose)
    return model, history


def signer_independent_eval(X, y, signers, class_names, epochs, batch_size, verbose):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report

    n_classes = len(class_names)
    test_signers = eligible_test_signers(y, signers, n_classes)
    train_only = sorted(set(signers) - set(test_signers))
    print(f"Held-out signers: {test_signers}" + (f" | train-only: {train_only}" if train_only else ""))

    fold_acc, all_true, all_pred, first_history = [], [], [], None
    for s in test_signers:
        te = signers == s
        X_tr, X_val, y_tr, y_val = train_test_split(
            X[~te], y[~te], test_size=0.15, stratify=y[~te], random_state=SEED)
        if verbose:
            print(f"\n--- fold: held-out signer {s} (n_test={int(te.sum())}) ---")
        model, history = train_one(X_tr, y_tr, X_val, y_val, n_classes, epochs, batch_size, verbose)
        first_history = first_history or (s, history)
        y_pred = model.predict(X[te], verbose=0).argmax(axis=1)
        acc = float((y_pred == y[te]).mean())
        fold_acc.append(acc)
        all_true.extend(y[te].tolist())
        all_pred.extend(y_pred.tolist())
        print(f"  fold test=signer {s}: accuracy={acc:.3f}")

    all_true, all_pred = np.array(all_true), np.array(all_pred)
    cm = confusion_matrix(all_true, all_pred, labels=range(n_classes))
    report = classification_report(all_true, all_pred, labels=range(n_classes),
                                   target_names=class_names, output_dict=True, zero_division=0)
    summary = {
        "test_signers": test_signers, "train_only_signers": train_only,
        "fold_accuracy": dict(zip(test_signers, fold_acc)),
        "mean_accuracy": float(np.mean(fold_acc)), "std_accuracy": float(np.std(fold_acc)),
        "pooled_accuracy": float((all_true == all_pred).mean()),
        "confusion_matrix": cm.tolist(), "per_class": report,
    }
    return summary, cm, first_history


def random_split_baseline(X, y, class_names, epochs, batch_size, verbose):
    from sklearn.model_selection import train_test_split

    n_classes = len(class_names)
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.1765, stratify=y_tmp,
                                                random_state=SEED)
    model, _ = train_one(X_tr, y_tr, X_val, y_val, n_classes, epochs, batch_size, verbose)
    acc = float((model.predict(X_te, verbose=0).argmax(axis=1) == y_te).mean())
    print(f"  random-split test accuracy: {acc:.3f} (n_test={len(y_te)})")
    return {"test_accuracy": acc, "n_test": int(len(y_te))}


def plot_confusion(cm, class_names, path, title):
    cm = np.array(cm)
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows != 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_curves(history, signer, path):
    h = history.history
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(h["accuracy"], label="train")
    a1.plot(h["val_accuracy"], label="val")
    a1.set_title(f"Accuracy (held-out signer {signer})")
    a1.set_xlabel("epoch")
    a1.legend()
    a2.plot(h["loss"], label="train")
    a2.plot(h["val_loss"], label="val")
    a2.set_title("Loss")
    a2.set_xlabel("epoch")
    a2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Train + evaluate the NSL BiLSTM pilot.")
    p.add_argument("--processed", default=C.PROCESSED_DIR)
    p.add_argument("--results", default=C.RESULTS_DIR)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2],
                   help="Keras fit verbosity: 0 silent, 1 progress bar, 2 one line/epoch")
    a = p.parse_args()

    import random
    import tensorflow as tf
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.makedirs(a.results, exist_ok=True)

    X, y, signers, class_names = load_processed(a.processed)
    print(f"X {X.shape} | classes {class_names} | signers {sorted(set(signers))}")
    print(f"Model params: {build_bilstm(X.shape[1], X.shape[2], len(class_names)).count_params():,}\n")

    print("=== Signer-independent (leave-one-signer-out) ===")
    si, cm, first_history = signer_independent_eval(
        X, y, signers, class_names, a.epochs, a.batch_size, a.verbose)
    print(f"\n  mean accuracy: {si['mean_accuracy']:.3f} +/- {si['std_accuracy']:.3f} "
          f"| pooled: {si['pooled_accuracy']:.3f}\n")
    print(f"  {'phrase':<18}{'prec':>7}{'recall':>8}{'f1':>7}{'support':>9}")
    for name in class_names:
        r = si["per_class"][name]
        print(f"  {name:<18}{r['precision']:>7.2f}{r['recall']:>8.2f}{r['f1-score']:>7.2f}{int(r['support']):>9}")

    print("\n=== Random-split baseline (stratified 70/15/15) ===")
    rnd = random_split_baseline(X, y, class_names, a.epochs, a.batch_size, a.verbose)

    plot_confusion(cm, class_names, os.path.join(a.results, "confusion_matrix_signer_indep.png"),
                   f"Signer-independent (mean acc {si['mean_accuracy']:.2f})")
    plot_curves(first_history[1], first_history[0], os.path.join(a.results, "training_curves.png"))
    with open(os.path.join(a.results, "metrics.json"), "w") as fh:
        json.dump({"config": {"epochs": a.epochs, "batch_size": a.batch_size,
                              "seq_len": int(X.shape[1]), "seed": SEED},
                   "class_names": class_names, "signer_independent": si,
                   "random_split_baseline": rnd}, fh, indent=2)
    print(f"\nSaved metrics.json + 2 PNGs to {a.results}")


if __name__ == "__main__":
    main()
