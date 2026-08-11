"""Compare BiLSTM architecture variants via leave-one-signer-out CV, to justify
the layer sizes picked in nslr/model.py rather than assert them.

Read-only with respect to the rest of the pipeline: this script does not import
or modify nslr/model.py's build_bilstm, and does not touch scripts/train_eval.py
or any file under nslr/. It defines its own local model builder (a superset of
build_bilstm's shape, parameterized) and its own local copy of the LOSO loop,
reusing only the data-loading helpers in nslr.dataset (eligible_test_signers,
load_processed) which are read-only lookups.

Runs the SAME protocol as scripts/train_eval.py (leave-one-signer-out, same
EarlyStopping/ReduceLROnPlateau recipe, same SEED) for every candidate
architecture, then writes:
    results/architecture_sweep.csv     one row per candidate
    results/architecture_sweep.md      the same table, markdown, for the report

    python -u scripts/sweep_architectures.py                 # all candidates
    python -u scripts/sweep_architectures.py --epochs 60      # quicker smoke test
"""

import argparse
import csv
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nslr import config as C
from nslr.dataset import eligible_test_signers, load_processed

SEED = 42


def build_candidate(seq_len, n_features, n_classes, lstm1, lstm2, dense, dropout, rnn="lstm"):
    """Local, parameterized stand-in for nslr.model.build_bilstm. lstm1/lstm2/dense
    at (64, 32, 32) with rnn="lstm" reproduces build_bilstm's architecture exactly
    (the "current" candidate below), so the sweep measures the shipped model on
    equal footing with the alternatives instead of approximating it."""
    from tensorflow import keras
    from tensorflow.keras import layers

    cell = layers.LSTM if rnn == "lstm" else layers.GRU
    layer_list = [
        keras.Input(shape=(seq_len, n_features)),
        layers.Masking(mask_value=0.0),
        layers.Bidirectional(cell(lstm1, return_sequences=bool(lstm2))),
        layers.Dropout(dropout),
    ]
    if lstm2:
        layer_list.append(layers.Bidirectional(cell(lstm2)))
        layer_list.append(layers.Dropout(dropout))
    if dense:
        layer_list.append(layers.Dense(dense, activation="relu"))
    layer_list.append(layers.Dense(n_classes, activation="softmax"))

    model = keras.Sequential(layer_list)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# name -> kwargs for build_candidate (minus seq_len/n_features/n_classes).
# "current" reproduces nslr.model.build_bilstm's fixed architecture exactly.
CANDIDATES = {
    "smaller":        dict(lstm1=32, lstm2=16, dense=16, dropout=0.3, rnn="lstm"),
    "current":        dict(lstm1=64, lstm2=32, dense=32, dropout=0.3, rnn="lstm"),
    "bigger":         dict(lstm1=128, lstm2=64, dense=64, dropout=0.3, rnn="lstm"),
    "single_layer":   dict(lstm1=64, lstm2=0, dense=32, dropout=0.3, rnn="lstm"),
    "no_dense_head":  dict(lstm1=64, lstm2=32, dense=0, dropout=0.3, rnn="lstm"),
    "high_dropout":   dict(lstm1=64, lstm2=32, dense=32, dropout=0.5, rnn="lstm"),
    "gru":            dict(lstm1=64, lstm2=32, dense=32, dropout=0.3, rnn="gru"),
}


def train_one(build_fn, X_tr, y_tr, X_val, y_val, n_classes, epochs, batch_size, verbose):
    from tensorflow import keras
    from sklearn.utils.class_weight import compute_class_weight

    present = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=present, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(present, weights)}

    model = build_fn(X_tr.shape[1], X_tr.shape[2], n_classes)
    # Same recipe as scripts/train_eval.py's train_one, so accuracy differences
    # across candidates reflect architecture, not a different training regime.
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                       min_delta=1e-3, restore_best_weights=True)
    lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                           patience=6, min_lr=1e-5)
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, class_weight=class_weight,
                        callbacks=[es, lr], verbose=verbose)
    return model, history


def loso_eval(build_fn, X, y, signers, n_classes, epochs, batch_size, verbose):
    from sklearn.model_selection import train_test_split

    test_signers = eligible_test_signers(y, signers, n_classes)
    fold_acc, epochs_run = [], []
    for s in test_signers:
        te = signers == s
        X_tr, X_val, y_tr, y_val = train_test_split(
            X[~te], y[~te], test_size=0.15, stratify=y[~te], random_state=SEED)
        model, history = train_one(build_fn, X_tr, y_tr, X_val, y_val, n_classes,
                                   epochs, batch_size, verbose)
        y_pred = model.predict(X[te], verbose=0).argmax(axis=1)
        fold_acc.append(float((y_pred == y[te]).mean()))
        epochs_run.append(len(history.history["loss"]))
    return {
        "test_signers": test_signers,
        "fold_accuracy": dict(zip(test_signers, fold_acc)),
        "mean_accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc)),
        "mean_epochs": float(np.mean(epochs_run)),
    }


FIELDS = ["name", "lstm1", "lstm2", "dense", "dropout", "rnn", "n_params",
          "mean_accuracy", "std_accuracy", "mean_epochs"]


def write_results(rows, results_dir):
    """(Re)write the CSV + markdown table from whatever rows have completed so
    far. Called after every candidate, not just at the end, so a run that gets
    interrupted (Ctrl+C, closed terminal, crash) still leaves a usable partial
    table instead of nothing."""
    ranked = sorted(rows, key=lambda r: r["mean_accuracy"], reverse=True)

    csv_path = os.path.join(results_dir, "architecture_sweep.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ranked)

    md_path = os.path.join(results_dir, "architecture_sweep.md")
    with open(md_path, "w") as fh:
        fh.write("| Architecture | LSTM1 | LSTM2 | Dense | Dropout | Cell | Params | "
                 "Mean LOSO Acc | Std | Avg Epochs |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in ranked:
            marker = " (current)" if r["name"] == "current" else ""
            fh.write(f"| {r['name']}{marker} | {r['lstm1']} | {r['lstm2'] or '-'} | "
                     f"{r['dense'] or '-'} | {r['dropout']} | {r['rnn'].upper()} | "
                     f"{r['n_params']:,} | {r['mean_accuracy']:.3f} | {r['std_accuracy']:.3f} | "
                     f"{r['mean_epochs']:.0f} |\n")
    return csv_path, md_path


def main():
    p = argparse.ArgumentParser(description="LOSO comparison across BiLSTM architecture variants.")
    p.add_argument("--processed", default=C.PROCESSED_DIR)
    p.add_argument("--results", default=C.RESULTS_DIR)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--candidates", nargs="+", default=list(CANDIDATES),
                   help="subset of candidate names to run (default: all)")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--threads", type=int, default=0,
                   help="cap TensorFlow's intra-op thread pool (0 = let TF use every core). "
                        "Only useful when several sweeps share one machine, as "
                        "scripts/sweep_parallel.py does -- TF ignores OMP_NUM_THREADS and "
                        "TF_NUM_INTRAOP_THREADS, so the cap has to be set in-process.")
    a = p.parse_args()

    import random
    import tensorflow as tf
    # Must happen before any op runs, i.e. before load_processed/model building.
    if a.threads:
        tf.config.threading.set_intra_op_parallelism_threads(a.threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    if a.deterministic:
        tf.config.experimental.enable_op_determinism()
    os.makedirs(a.results, exist_ok=True)

    X, y, signers, class_names = load_processed(a.processed)
    n_classes = len(class_names)
    print(f"X {X.shape} | classes {class_names} | signers {sorted(set(signers))}")
    print(f"Candidates: {a.candidates}\n")

    rows = []
    t_start = time.perf_counter()
    for i, name in enumerate(a.candidates, 1):
        cfg = CANDIDATES[name]
        build_fn = lambda seq_len, n_features, nc, cfg=cfg: build_candidate(
            seq_len, n_features, nc, **cfg)
        n_params = build_fn(X.shape[1], X.shape[2], n_classes).count_params()
        print(f"[{i}/{len(a.candidates)}] {name} ({cfg}, {n_params:,} params) ...", flush=True)

        t0 = time.perf_counter()
        result = loso_eval(build_fn, X, y, signers, n_classes, a.epochs, a.batch_size, a.verbose)
        dt = time.perf_counter() - t0
        print(f"  -> mean {result['mean_accuracy']:.3f} +/- {result['std_accuracy']:.3f} "
              f"(avg {result['mean_epochs']:.0f} epochs, {dt/60:.1f} min)\n", flush=True)

        rows.append({
            "name": name, **cfg, "n_params": n_params,
            "mean_accuracy": result["mean_accuracy"], "std_accuracy": result["std_accuracy"],
            "mean_epochs": result["mean_epochs"],
        })
        csv_path, md_path = write_results(rows, a.results)
        print(f"  [saved {len(rows)}/{len(a.candidates)} candidates -> {csv_path}]\n", flush=True)

    total_min = (time.perf_counter() - t_start) / 60
    print(f"=== Ranked by mean LOSO accuracy ===")
    for r in sorted(rows, key=lambda r: r["mean_accuracy"], reverse=True):
        marker = " <-- current" if r["name"] == "current" else ""
        print(f"  {r['name']:<14} {r['mean_accuracy']:.3f} +/- {r['std_accuracy']:.3f}  "
              f"({r['n_params']:,} params){marker}")
    print(f"\nSaved {csv_path}\nSaved {md_path}\nTotal sweep time: {total_min:.1f} min")


if __name__ == "__main__":
    main()
