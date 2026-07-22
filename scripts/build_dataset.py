"""Compile raw clips into fixed-length normalized tensors (X/mask/y + metadata).

Run find_seq_len.py first so seq_len.json exists.

    python scripts/build_dataset.py
    python scripts/build_dataset.py --min-hand-detect 0.5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nslr import config as C
from nslr.dataset import compile_dataset


def main():
    p = argparse.ArgumentParser(description="Compile raw NSL clips into X/mask/y tensors.")
    p.add_argument("--data", default=C.RAW_DIR)
    p.add_argument("--out", default=C.PROCESSED_DIR)
    p.add_argument("--min-hand-detect", type=float, default=0.0,
                   help="drop clips whose any_hand_detect_rate is below this (0..1); default keeps all")
    a = p.parse_args()

    r = compile_dataset(a.data, a.out, a.min_hand_detect)
    X, mask, y, modes = r["X"], r["mask"], r["y"], r["modes"]

    print(f"\nSEQ_LEN {r['seq_len']} | X {X.shape} mask {mask.shape} y {y.shape}"
          + (f" | dropped {r['n_dropped']}" if r["n_dropped"] else ""))
    print(f"length: exact={modes['exact']} subsampled={modes['subsampled']} padded={modes['padded']}"
          f" | padding {1.0 - mask.mean():.1%}")
    print("per-class:", {n: int((y == lab).sum()) for n, lab in r["label_map"].items()})
    signers = sorted(set(row["signer"] for row in r["manifest"]))
    print("per-signer:", {s: sum(1 for row in r["manifest"] if row["signer"] == s) for s in signers})
    print("saved to", r["out_dir"])


if __name__ == "__main__":
    main()
