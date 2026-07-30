"""Build processed tensors from raw clips, and load them back for training."""

import csv
import json
import os

import numpy as np

from . import config as C
from .preprocess import load_seq_len, normalize_clip, standardize_length


def compile_dataset(data_path=C.RAW_DIR, out_dir=C.PROCESSED_DIR,
                    min_hand_detect=0.0, seq_len=None):
    """Normalize + standardize every raw clip and save X/mask/y + metadata.
    Returns a summary dict. seq_len defaults to <out_dir>/seq_len.json."""
    os.makedirs(out_dir, exist_ok=True)
    if seq_len is None:
        seq_len = load_seq_len(out_dir)

    # Only classes that actually contain clips — an empty folder (e.g. a
    # not-yet-recorded 'none' class) must not create a zero-sample label.
    class_names = sorted(
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
        and any(f.endswith(".npy") for f in os.listdir(os.path.join(data_path, d)))
    )
    if not class_names:
        raise SystemExit(f"No non-empty class folders under {data_path}")
    label_map = {name: i for i, name in enumerate(class_names)}

    X, masks, y, manifest, dropped = [], [], [], [], []
    modes = {"exact": 0, "subsampled": 0, "padded": 0}

    for name in class_names:
        cdir = os.path.join(data_path, name)
        for fname in sorted(f for f in os.listdir(cdir) if f.endswith(".npy")):
            fpath = os.path.join(cdir, fname)
            signer = fname.split("__")[1] if "__" in fname else ""

            if min_hand_detect > 0.0:
                any_hand = _read_any_hand(fpath[:-4] + ".json")
                if any_hand is not None and any_hand < min_hand_detect:
                    dropped.append({"class": name, "signer": signer, "source_file": fpath,
                                    "any_hand_detect_rate": any_hand,
                                    "reason": f"any_hand<{min_hand_detect}"})
                    continue

            raw = np.load(fpath).astype(np.float32)
            if raw.ndim != 2 or raw.shape[1] != C.FEATURE_DIM:
                dropped.append({"class": name, "signer": signer, "source_file": fpath,
                                "any_hand_detect_rate": "", "reason": f"bad_shape={raw.shape}"})
                continue

            fixed, mask, mode = standardize_length(normalize_clip(raw), seq_len)
            modes[mode] += 1
            X.append(fixed)
            masks.append(mask)
            y.append(label_map[name])
            manifest.append({"index": len(X) - 1, "class": name, "signer": signer,
                             "source_file": fpath, "n_frames_raw": raw.shape[0],
                             "standardize": mode})

    if not X:
        raise SystemExit("No clips survived — check data_path and min_hand_detect.")

    X = np.stack(X).astype(np.float32)
    mask_arr = np.stack(masks).astype(bool)
    y = np.array(y, dtype=np.int64)

    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "mask.npy"), mask_arr)
    np.save(os.path.join(out_dir, "y.npy"), y)
    with open(os.path.join(out_dir, "label_map.json"), "w") as fh:
        json.dump(label_map, fh, indent=2)
    _write_csv(os.path.join(out_dir, "manifest.csv"),
               ["index", "class", "signer", "source_file", "n_frames_raw", "standardize"],
               manifest)
    if dropped:
        _write_csv(os.path.join(out_dir, "dropped.csv"),
                   ["class", "signer", "source_file", "any_hand_detect_rate", "reason"],
                   dropped)

    return {"X": X, "mask": mask_arr, "y": y, "label_map": label_map, "seq_len": seq_len,
            "modes": modes, "n_dropped": len(dropped), "manifest": manifest, "out_dir": out_dir}


def load_processed(processed_dir=C.PROCESSED_DIR):
    """Return (X, y, signers, class_names); signers[i] index-aligned to X/y."""
    X = np.load(os.path.join(processed_dir, "X.npy"))
    y = np.load(os.path.join(processed_dir, "y.npy"))
    with open(os.path.join(processed_dir, "label_map.json")) as fh:
        label_map = json.load(fh)
    class_names = [n for n, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    signers = [""] * len(y)
    with open(os.path.join(processed_dir, "manifest.csv")) as fh:
        for row in csv.DictReader(fh):
            signers[int(row["index"])] = row["signer"]
    signers = np.array(signers)

    assert len(X) == len(y) == len(signers), "X / y / manifest length mismatch"
    return X, y, signers, class_names


def eligible_test_signers(y, signers, n_classes):
    """Signers that can be held out for a leave-one-signer-out fold: removing the
    signer must still leave EVERY class present in the training remainder,
    otherwise the model can't learn a class it will be scored on. (A class held
    by only one signer therefore can't be tested signer-independently.)"""
    all_labels = set(range(n_classes))
    return [s for s in sorted(set(signers))
            if set(y[signers != s].tolist()) == all_labels]


def _read_any_hand(json_path):
    try:
        with open(json_path) as fh:
            return json.load(fh).get("any_hand_detect_rate")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write_csv(path, fields, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
