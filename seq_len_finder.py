import os
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = r"D:\College notes\SEM-6\Minor Project\NSL\Data Collection\Data\raw"


def load_all_frame_counts():
    """Walk DATA_PATH and collect every clip's frame count."""
    frame_counts = []
    for class_name in os.listdir(DATA_PATH):
        class_path = os.path.join(DATA_PATH, class_name)
        if not os.path.isdir(class_path):
            continue

        npy_files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
        for fname in npy_files:
            fpath = os.path.join(class_path, fname)
            data = np.load(fpath)
            frame_counts.append(data.shape[0])
    return frame_counts


def main():
    frame_counts = load_all_frame_counts()
    print(f"Loaded {len(frame_counts)} clips.\n")

    p95 = np.percentile(frame_counts, 95)

    print("--- Results ---")
    print(f"Min frames      : {np.min(frame_counts)}")
    print(f"Mean frames     : {np.mean(frame_counts):.1f}")
    print(f"Max frames      : {np.max(frame_counts)}")
    print(f"95th percentile : {p95:.1f}")
    print(f"\nRecommended SEQ_LEN : {int(round(p95))}")

    plt.figure(figsize=(8, 5))
    plt.hist(frame_counts, bins=30, color="#55A868", edgecolor="black", alpha=0.8)
    plt.axvline(p95, color="green", linestyle="--", label=f"95th pct = {p95:.1f}")
    plt.xlabel("frame count (per clip)")
    plt.ylabel("number of clips")
    plt.title(f"Frame-count distribution — all {len(frame_counts)} clips")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
