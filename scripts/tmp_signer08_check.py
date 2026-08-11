"""TEMPORARY / throwaway script -- check whether replacing signer08's two
anomalous help_danger clips (007, 010; see data/raw/help_danger/*.json
'duplicated_from' fields) moved signer08's leave-one-signer-out accuracy.

Non-destructive: rebuilds a dataset from the CURRENT data/raw/ into a scratch
directory (does not touch data/processed/), then runs exactly ONE LOSO fold
(held-out signer = signer08) using the same build_bilstm architecture and the
same train_one recipe as scripts/train_eval.py.

    python -u scripts/tmp_signer08_check.py
"""

import os
import random
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nslr import config as C
from nslr.dataset import compile_dataset
from nslr.model import build_bilstm

SEED = 42
SCRATCH_OUT = os.path.join(C.REPO_ROOT, "results", "_tmp_signer08_check_processed")


def train_one(X_tr, y_tr, X_val, y_val, n_classes, epochs, batch_size, verbose):
    from tensorflow import keras
    from sklearn.utils.class_weight import compute_class_weight

    present = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=present, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(present, weights)}

    model = build_bilstm(X_tr.shape[1], X_tr.shape[2], n_classes)
    # Same recipe as scripts/train_eval.py's train_one.
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                       min_delta=1e-3, restore_best_weights=True)
    lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                           patience=6, min_lr=1e-5)
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, class_weight=class_weight,
                        callbacks=[es, lr], verbose=verbose)
    return model, history


def main():
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    print(f"Rebuilding dataset from {C.RAW_DIR} into scratch dir {SCRATCH_OUT} "
          f"(data/processed/ is untouched) ...")
    r = compile_dataset(C.RAW_DIR, SCRATCH_OUT, seq_len=137)
    X, y = r["X"], r["y"]
    with open(os.path.join(SCRATCH_OUT, "manifest.csv")) as fh:
        import csv
        signers = [""] * len(y)
        for row in csv.DictReader(fh):
            signers[int(row["index"])] = row["signer"]
    signers = np.array(signers)
    class_names = [n for n, _ in sorted(r["label_map"].items(), key=lambda kv: kv[1])]
    n_classes = len(class_names)
    print(f"X {X.shape} | classes {class_names}\n")

    target = "signer08"
    te = signers == target
    if not te.any():
        raise SystemExit(f"No clips found for {target} -- check the signer name.")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X[~te], y[~te], test_size=0.15, stratify=y[~te], random_state=SEED)
    print(f"Held out {target}: train {len(X_tr)} / val {len(X_val)} / test {int(te.sum())}")

    model, history = train_one(X_tr, y_tr, X_val, y_val, n_classes,
                               epochs=200, batch_size=16, verbose=2)
    y_pred = model.predict(X[te], verbose=0).argmax(axis=1)
    acc = float((y_pred == y[te]).mean())

    print(f"\n{target} LOSO accuracy: {acc:.3f}  "
         f"(over {len(history.history['loss'])} epochs)")
    print("Compare against the original metrics.json value: 0.657")


if __name__ == "__main__":
    main()