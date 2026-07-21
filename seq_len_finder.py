import os
import random
import numpy as np

DATA_PATH = r"C:\path\to\your\data"  # change this to your actual path

frame_counts = []

for class_name in os.listdir(DATA_PATH):
    class_path = os.path.join(DATA_PATH, class_name)
    if not os.path.isdir(class_path):
        continue

    # get all .npy files in this class folder
    npy_files = [f for f in os.listdir(class_path) if f.endswith('.npy')]

    if len(npy_files) < 2:
        print(f"Warning: {class_name} has fewer than 2 .npy files, taking what's available")

    # pick 2 randomly
    sampled = random.sample(npy_files, min(2, len(npy_files)))

    for fname in sampled:
        fpath = os.path.join(class_path, fname)
        data = np.load(fpath)
        frame_count = data.shape[0]
        frame_counts.append(frame_count)
        print(f"{class_name} | {fname} | frames: {frame_count} | shape: {data.shape}")

print("\n--- Results ---")
print(f"Total clips sampled : {len(frame_counts)}")
print(f"Min frames          : {np.min(frame_counts)}")
print(f"Mean frames         : {np.mean(frame_counts):.1f}")
print(f"Max frames          : {np.max(frame_counts)}")
print(f"95th percentile     : {np.percentile(frame_counts, 95):.1f}")
print(f"\nRecommended SEQ_LEN : {int(np.percentile(frame_counts, 95))}")