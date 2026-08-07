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
import time

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

    present = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=present, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(present, weights)}

    model = build_bilstm(X_tr.shape[1], X_tr.shape[2], n_classes)
    # min_delta is the load-bearing argument. At the default 0.0, ANY improvement
    # resets the patience counter -- including 0.01000 -> 0.00999. Once val_loss
    # is asymptoting toward zero it keeps setting microscopic new records, so the
    # run never ends on convergence; it ends when an Adam instability spike
    # finally breaks the streak. That made fold length (24 to 113 epochs) a
    # measure of when a random hiccup happened rather than of anything real.
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                       min_delta=1e-3, restore_best_weights=True)
    # Those spikes are Adam taking oversized steps once gradients get tiny near
    # the minimum. Decaying the LR on a plateau removes the cause instead of
    # tolerating it. patience must stay well below EarlyStopping's, or the run
    # dies before the lower LR gets a chance to help.
    lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                           patience=6, min_lr=1e-5)
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, class_weight=class_weight,
                        callbacks=[es, lr], verbose=verbose)
    return model, history


def signer_independent_eval(X, y, signers, class_names, epochs, batch_size, verbose):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report

    n_classes = len(class_names)
    test_signers = eligible_test_signers(y, signers, n_classes)
    train_only = sorted(set(signers) - set(test_signers))
    print(f"Held-out signers: {test_signers}" + (f" | train-only: {train_only}" if train_only else ""))

    fold_acc, all_true, all_pred, histories = [], [], [], []
    n_folds = len(test_signers)
    t_start = time.perf_counter()
    for i, s in enumerate(test_signers, 1):
        te = signers == s
        X_tr, X_val, y_tr, y_val = train_test_split(
            X[~te], y[~te], test_size=0.15, stratify=y[~te], random_state=SEED)
        print(f"  [{i}/{n_folds}] training, held-out {s} "
              f"(train {len(X_tr)} / val {len(X_val)} / test {int(te.sum())}) ...", flush=True)
        t0 = time.perf_counter()
        model, history = train_one(X_tr, y_tr, X_val, y_val, n_classes, epochs, batch_size, verbose)
        y_pred = model.predict(X[te], verbose=0).argmax(axis=1)
        acc = float((y_pred == y[te]).mean())
        fold_acc.append(acc)
        # The plain dict, not the History object — History holds a reference to
        # its model, and keeping eight of those alive defeats the point of
        # discarding each fold's model as soon as it has been scored.
        histories.append((s, dict(history.history), acc))
        all_true.extend(y[te].tolist())
        all_pred.extend(y_pred.tolist())
        dt, elapsed = time.perf_counter() - t0, time.perf_counter() - t_start
        eta = elapsed / i * (n_folds - i)
        print(f"  [{i}/{n_folds}] {s}: accuracy={acc:.3f}  "
              f"({len(history.history['loss'])} epochs, {dt:.0f}s)"
              + (f"   ~{eta/60:.1f} min left" if i < n_folds else ""), flush=True)

    all_true, all_pred = np.array(all_true), np.array(all_pred)
    cm = confusion_matrix(all_true, all_pred, labels=range(n_classes))
    report = classification_report(all_true, all_pred, labels=range(n_classes),
                                   target_names=class_names, output_dict=True, zero_division=0)
    summary = {
        "test_signers": test_signers, "train_only_signers": train_only,
        "fold_accuracy": dict(zip(test_signers, fold_acc)),
        "mean_accuracy": float(np.mean(fold_acc)), "std_accuracy": float(np.std(fold_acc)),
        "pooled_accuracy": float((all_true == all_pred).mean()),
        # Where early stopping landed per fold. Makes "every fold converged the
        # same way" a number in the metrics file, not just an eyeball call on
        # the curves — and shows whether `--epochs` was ever the binding limit.
        "epochs_run": {s: len(h["loss"]) for s, h, _ in histories},
        "confusion_matrix": cm.tolist(), "per_class": report,
    }
    return summary, cm, histories


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


def plot_fold_grid(histories, path, metric, suptitle):
    """One panel per leave-one-signer-out fold.

    This used to plot a single fold, which showed that *one* of the eight models
    converged cleanly and left the other seven unaccounted for. A grid makes
    "every fold behaved the same way" something a reader can check instead of
    something the report asserts from a sample of one.

    On the accuracy grid each panel also carries a dashed line at that fold's
    held-out score. That line is the protocol made visible: train and val are
    both drawn from the signers *inside* training, so they sit high together;
    the dashed line is the same model meeting a signer it has never seen. The
    gap between them is the cost of a stranger, per fold.
    """
    n = len(histories)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.9 * rows),
                             squeeze=False, sharex=True, sharey=True)
    flat = axes.ravel()

    for ax, (signer, hist, acc) in zip(flat, histories):
        ax.plot(hist[metric], label="train", lw=1.3)
        ax.plot(hist[f"val_{metric}"], label="val", lw=1.3)
        if metric == "accuracy":
            ax.axhline(acc, color="crimson", ls="--", lw=1.2, label="held-out signer")
        ax.set_title(f"{signer} — test {acc:.3f} — {len(hist[metric])} ep", fontsize=9)
        ax.grid(alpha=0.25)
    for ax in flat[n:]:
        ax.axis("off")

    # sharex hides tick labels on every row but the last, so a column whose
    # bottom panel was blanked would lose its axis entirely. Label the lowest
    # panel that is actually drawn in each column.
    for c in range(cols):
        drawn = [axes[r][c] for r in range(rows) if r * cols + c < n]
        if drawn:
            drawn[-1].set_xlabel("epoch")
            drawn[-1].tick_params(labelbottom=True)
    for row in axes:
        row[0].set_ylabel(metric)

    # One legend for the whole figure. Per-axes it lands on top of the curves —
    # every panel is a plateau with no empty corner to put it in.
    handles, labels = flat[0].get_legend_handles_labels()
    fig.suptitle(suptitle, fontsize=11, y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
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
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                   help="reproducible runs. SEED alone does not achieve this: TF's parallel "
                        "CPU reductions vary between runs, which moved per-fold accuracy by "
                        "up to 2.7 points on identical data. Costs some speed.")
    a = p.parse_args()

    import random
    import tensorflow as tf
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    if a.deterministic:
        tf.config.experimental.enable_op_determinism()
    os.makedirs(a.results, exist_ok=True)

    X, y, signers, class_names = load_processed(a.processed)
    print(f"X {X.shape} | classes {class_names} | signers {sorted(set(signers))}")
    print(f"Model params: {build_bilstm(X.shape[1], X.shape[2], len(class_names)).count_params():,}\n")

    print("=== Signer-independent (leave-one-signer-out) ===")
    si, cm, histories = signer_independent_eval(
        X, y, signers, class_names, a.epochs, a.batch_size, a.verbose)
    print(f"\n  mean accuracy: {si['mean_accuracy']:.3f} +/- {si['std_accuracy']:.3f} "
          f"| pooled: {si['pooled_accuracy']:.3f}\n")
    print(f"  {'phrase':<18}{'prec':>7}{'recall':>8}{'f1':>7}{'support':>9}")
    for name in class_names:
        r = si["per_class"][name]
        print(f"  {name:<18}{r['precision']:>7.2f}{r['recall']:>8.2f}{r['f1-score']:>7.2f}{int(r['support']):>9}")

    print("\n=== Random-split baseline (stratified 70/15/15) ===")
    rnd = random_split_baseline(X, y, class_names, a.epochs, a.batch_size, a.verbose)

    # Pooled, not mean-of-folds: the matrix IS the pooled predictions, so a mean
    # in the title invites anyone who divides the diagonal by the total to get a
    # different number and wonder which is wrong.
    plot_confusion(cm, class_names, os.path.join(a.results, "confusion_matrix_signer_indep.png"),
                   f"Signer-independent — pooled over {len(si['test_signers'])} folds "
                   f"({si['pooled_accuracy']:.3f})")
    plot_fold_grid(histories, os.path.join(a.results, "training_curves.png"), "accuracy",
                   f"Accuracy per fold — train/val are signers INSIDE training; "
                   f"dashed = the held-out signer (pooled {si['pooled_accuracy']:.3f})")
    plot_fold_grid(histories, os.path.join(a.results, "training_loss_curves.png"), "loss",
                   "Loss per fold — val_loss is what EarlyStopping monitors")
    with open(os.path.join(a.results, "metrics.json"), "w") as fh:
        json.dump({"config": {"epochs": a.epochs, "batch_size": a.batch_size,
                              "seq_len": int(X.shape[1]), "seed": SEED},
                   "class_names": class_names, "signer_independent": si,
                   "random_split_baseline": rnd}, fh, indent=2)
    print(f"\nSaved metrics.json + 3 PNGs to {a.results}")


if __name__ == "__main__":
    main()
