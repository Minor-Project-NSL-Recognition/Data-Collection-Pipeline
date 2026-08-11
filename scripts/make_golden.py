"""Generate the golden fixtures that pin the Flutter app's Dart port to this pipeline.

The offline app re-implements `nslr/preprocess.py` and `nslr/ood.py` in Dart. If the
two ever disagree, nothing throws -- the BiLSTM is simply handed features it was
never trained on and answers confidently anyway. There is no runtime symptom, so
the agreement is pinned by test instead: this script writes real clips and real
gate values through the Python path, and `app/test/preprocess_test.dart` replays
them through the Dart path.

Run it after ANY of these change:
    nslr/preprocess.py · nslr/ood.py · nslr/config.py · seq_len · the model

    python scripts/make_golden.py
    cd app && flutter test

Covers all three branches of standardize_length (padded / subsampled / exact) plus
the Mahalanobis distances, including exact-prototype embeddings where the distance
must be ~0 (a transposed precision matrix or a sign slip cannot survive that).
"""

import argparse
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nslr import config as C
from nslr import ood
from nslr.preprocess import normalize_clip, standardize_length

DEFAULT_OUT = os.path.join(C.REPO_ROOT, "app", "test", "fixtures",
                           "preprocess_golden.json")


def pick_clips(raw_dir, seq_len):
    """One real clip per standardize branch, smallest of each to keep the fixture small."""
    found = {}
    for cls in sorted(os.listdir(raw_dir)):
        cdir = os.path.join(raw_dir, cls)
        if not os.path.isdir(cdir):
            continue
        for fname in sorted(f for f in os.listdir(cdir) if f.endswith(".npy")):
            arr = np.load(os.path.join(cdir, fname)).astype(np.float32)
            if arr.ndim != 2 or arr.shape[1] != C.FEATURE_DIM:
                continue
            n = arr.shape[0]
            kind = "padded" if n < seq_len else ("subsampled" if n > seq_len else "exact")
            if kind not in found or n < found[kind][1].shape[0]:
                found[kind] = (f"{cls}/{fname}", arr)
    return found


def main():
    p = argparse.ArgumentParser(description="Write Dart<->Python golden fixtures.")
    p.add_argument("--data", default=C.RAW_DIR)
    p.add_argument("--results", default=C.RESULTS_DIR)
    p.add_argument("--out", default=DEFAULT_OUT)
    a = p.parse_args()

    meta_path = os.path.join(a.results, "model_meta.json")
    ood_path = os.path.join(a.results, "ood.json")
    for path in (meta_path, ood_path):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} — run train_model.py then export_tflite.py")

    with open(meta_path) as fh:
        meta = json.load(fh)
    seq_len = int(meta["seq_len"])
    print(f"seq_len {seq_len} | {len(meta['class_names'])} classes")

    cases = []
    for kind, (name, raw) in sorted(pick_clips(a.data, seq_len).items()):
        # Trim very long clips: the subsample branch only needs n > seq_len, and a
        # 400-frame clip would bloat the fixture for no extra coverage.
        if raw.shape[0] > seq_len + 40:
            raw = raw[: seq_len + 40]
            kind = "subsampled"
        fixed, mask, mode = standardize_length(normalize_clip(raw.copy()), seq_len)
        cases.append({
            "name": name,
            "expected_mode": mode,
            "n_frames": int(raw.shape[0]),
            "raw": raw.astype(np.float32).tolist(),
            "expected": fixed.astype(np.float32).tolist(),
            "real_frames_in_mask": int(mask.sum()),
        })
        print(f"  {mode:11s} {name}  {raw.shape} -> {fixed.shape}")

    with open(ood_path) as fh:
        gate = json.load(fh)
    means = np.array(gate["means"])
    precision = np.array(gate["precision"])

    # Alternate exact prototypes with displaced ones, so "distance 0" and
    # "distance > 0" are both asserted.
    rng = np.random.default_rng(7)
    embeddings = []
    for c in range(means.shape[0]):
        embeddings.append(means[c])
        embeddings.append(means[c] + rng.normal(0, 0.5, means.shape[1]))
    embeddings = np.stack(embeddings)
    dists, _ = ood.mahalanobis_min(embeddings, means, precision)

    payload = {
        "seq_len": seq_len,
        "feature_dim": C.FEATURE_DIM,
        "class_names": meta["class_names"],
        "cases": cases,
        "ood": {
            "threshold": float(gate["threshold"]),
            "embeddings": embeddings.astype(np.float64).tolist(),
            "expected_distances": dists.astype(np.float64).tolist(),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(payload, fh)
    print(f"\nwrote {a.out}  ({os.path.getsize(a.out)/1024/1024:.1f} MB)")
    print(f"  {len(cases)} preprocess cases, {len(embeddings)} ood cases")
    print("\nNow run:  cd app ; flutter test")


if __name__ == "__main__":
    main()
