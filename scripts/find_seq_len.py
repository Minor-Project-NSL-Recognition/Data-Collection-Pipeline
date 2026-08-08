"""Derive SEQ_LEN from the raw clip frame-count distribution.

Writes <out>/seq_len.json (read by build_dataset.py) and a histogram PNG.

    python scripts/find_seq_len.py
    python scripts/find_seq_len.py --percentile 95 --show
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from nslr import config as C


def frame_counts(data_path):
    counts = []

    for cls in sorted(os.listdir(data_path)):
#  this lists the directories insidied the raw data folder 
        cdir = os.path.join(data_path, cls)
# this create the abs path of each of the directories for the 
        if not os.path.isdir(cdir):
            continue
        for f in os.listdir(cdir):
            if f.endswith(".npy"):
                counts.append(int(np.load(os.path.join(cdir, f), mmap_mode="r").shape[0]))
    return np.array(counts)


def main():
    p = argparse.ArgumentParser(description="Derive SEQ_LEN from raw clip frame counts.")
    p.add_argument("--data", default=C.RAW_DIR)
    p.add_argument("--out", default=C.PROCESSED_DIR)
    p.add_argument("--percentile", type=float, default=95.0)
    p.add_argument("--show", action="store_true", help="open the histogram window (default: save only)")
    a = p.parse_args()

    counts = frame_counts(a.data)
    if len(counts) == 0:
        raise SystemExit(f"No .npy clips under {a.data}")

    pct = float(np.percentile(counts, a.percentile))
    # this calculates the actual 95th percentile of the distribution of the frame counts of the file as  a.percentile=95.
     
    seq_len = int(round(pct))
    # rounding off the seq_len

    print(f"{len(counts)} clips | min={counts.min()} mean={counts.mean():.1f} "
          f"median={np.median(counts):.0f} max={counts.max()} | p{a.percentile:g}={pct:.1f}")
    print(f"SEQ_LEN = {seq_len}")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "seq_len.json"), "w") as fh:
        json.dump({"seq_len": seq_len, "percentile": a.percentile, "n_clips": int(len(counts)),
                   "min": int(counts.min()), "mean": round(float(counts.mean()), 1),
                   "median": int(np.median(counts)), "max": int(counts.max())}, fh, indent=2)

    plt.figure(figsize=(8, 5))
    plt.hist(counts, bins=30, color="#55A868", edgecolor="black", alpha=0.8)
    plt.axvline(pct, color="green", linestyle="--",
                label=f"p{a.percentile:g}={pct:.1f} -> SEQ_LEN {seq_len}")
    plt.xlabel("frames per clip")
    plt.ylabel("clips")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "seq_len_hist.png"), dpi=120)
    print("wrote seq_len.json + seq_len_hist.png")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
