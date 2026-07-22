"""
NSL Emergency-Phrase Recognition — Landmark Processing Step
=============================================================

Compiles the raw, variable-length clips produced by record.py into fixed-length,
normalized tensors ready for BiLSTM training.

Pipeline per clip (in this order):
    1. Load the raw (n_frames, 225) array.
    2. Normalize EVERY frame:
         - pose block   (99 values) -> shoulder-midpoint anchor
         - left hand    (63 values) -> wrist anchor
         - right hand   (63 values) -> wrist anchor
       Normalization must happen BEFORE padding: a padded/undetected block is
       already all-zero, and the anchor formulas map an all-zero block to
       itself (0 - 0 = 0), so padding stays exactly zero after normalization
       and is trivially maskable.
    3. Pad or clip to SEQ_LEN:
         - longer than SEQ_LEN  -> keep the FIRST SEQ_LEN frames (the sign
           starts at frame 0; a slow/late finish is more disposable than the
           onset).
         - shorter than SEQ_LEN -> zero-pad at the END (never loop/duplicate
           real frames — that fabricates motion that never happened).
    4. Record a per-frame mask (1 = real frame, 0 = padding), so the BiLSTM
       can be told (via a Masking layer / packed sequence) to ignore padding
       instead of learning from it.

Output (<output_dir>/processed/):
    X.npy          float32 (n_clips, SEQ_LEN, 225)   normalized, padded/clipped landmarks
    mask.npy       bool    (n_clips, SEQ_LEN)         True where a frame is real, False where padded
    y.npy          int64   (n_clips,)                 integer class label per clip
    label_map.json {class_name: label_int}            so label ints can be decoded later
    manifest.csv   one row per clip: index, class, signer, source .npy path, n_frames_raw

Requirements:
    pip install numpy
"""

import csv
import json
import os

import numpy as np

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DATA_PATH = r"D:\College notes\SEM-6\Minor Project\NSL\Data Collection\Data\raw"
OUTPUT_DIR = r"D:\College notes\SEM-6\Minor Project\NSL\Data Collection\Data\processed"

SEQ_LEN = 139   # from seq_len_finder.py — 95th percentile of raw frame counts

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
POSE_DIM = POSE_LANDMARKS * 3   # 99
HAND_DIM = HAND_LANDMARKS * 3   # 63
FEATURE_DIM = POSE_DIM + HAND_DIM * 2   # 225

# Slice boundaries within the 225-length feature vector (must match record.py's
# concatenation order: pose -> left hand -> right hand).
POSE_SLICE = slice(0, POSE_DIM)
LEFT_HAND_SLICE = slice(POSE_DIM, POSE_DIM + HAND_DIM)
RIGHT_HAND_SLICE = slice(POSE_DIM + HAND_DIM, FEATURE_DIM)

# Landmark indices used as normalization anchors (MediaPipe convention).
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
HAND_WRIST = 0
HAND_MIDDLE_FINGER_MCP = 9

EPS = 1e-6   # avoids division by zero when an anchor pair collapses (e.g. all-zero block)


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

def normalize_hand_block(flat_block):
    """Wrist-anchor normalize one hand's (21, 3) block, given as a flat (63,) array.

    h_hat_i = (h_i - h_0) / (||h_9 - h_0|| + eps),  i = 0..20

    An all-zero block (hand not detected) maps to itself: h_0 = 0, so every
    numerator is 0 regardless of the denominator.
    """
    pts = flat_block.reshape(HAND_LANDMARKS, 3)
    wrist = pts[HAND_WRIST]
    scale = np.linalg.norm(pts[HAND_MIDDLE_FINGER_MCP] - wrist) + EPS
    normalized = (pts - wrist) / scale
    return normalized.reshape(-1)


def normalize_pose_block(flat_block):
    """Shoulder-midpoint anchor normalize the (33, 3) pose block, given as a flat (99,) array.

    p_hat_i = (p_i - 0.5*(p_11 + p_12)) / (||p_11 - p_12|| + eps),  i = 0..32
    """
    pts = flat_block.reshape(POSE_LANDMARKS, 3)
    left_shoulder = pts[POSE_LEFT_SHOULDER]
    right_shoulder = pts[POSE_RIGHT_SHOULDER]
    mid = 0.5 * (left_shoulder + right_shoulder)
    scale = np.linalg.norm(left_shoulder - right_shoulder) + EPS
    normalized = (pts - mid) / scale
    return normalized.reshape(-1)


def normalize_clip(clip):
    """Normalize every frame of a (n_frames, 225) clip. Returns a new array
    of the same shape; the three feature blocks are normalized independently."""
    out = np.empty_like(clip)
    for t in range(clip.shape[0]):
        out[t, POSE_SLICE] = normalize_pose_block(clip[t, POSE_SLICE])
        out[t, LEFT_HAND_SLICE] = normalize_hand_block(clip[t, LEFT_HAND_SLICE])
        out[t, RIGHT_HAND_SLICE] = normalize_hand_block(clip[t, RIGHT_HAND_SLICE])
    return out


# --------------------------------------------------------------------------
# Pad / clip
# --------------------------------------------------------------------------

def pad_or_clip(clip, seq_len):
    """Return (fixed_clip, mask) where fixed_clip has shape (seq_len, 225).

    Longer clips keep their FIRST seq_len frames (the sign's onset).
    Shorter clips are zero-padded at the END; mask is False on those frames.
    """
    n_frames = clip.shape[0]
    mask = np.zeros(seq_len, dtype=bool)

    if n_frames >= seq_len:
        fixed = clip[:seq_len]
        mask[:] = True
    else:
        fixed = np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)
        fixed[:n_frames] = clip
        mask[:n_frames] = True

    return fixed, mask


# --------------------------------------------------------------------------
# Compilation
# --------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    class_names = sorted(
        d for d in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, d))
    )
    label_map = {name: i for i, name in enumerate(class_names)}

    X, mask_list, y, manifest_rows = [], [], [], []

    for class_name in class_names:
        class_path = os.path.join(DATA_PATH, class_name)
        npy_files = sorted(f for f in os.listdir(class_path) if f.endswith(".npy"))

        for fname in npy_files:
            fpath = os.path.join(class_path, fname)
            raw = np.load(fpath).astype(np.float32)

            normalized = normalize_clip(raw)
            fixed, mask = pad_or_clip(normalized, SEQ_LEN)

            X.append(fixed)
            mask_list.append(mask)
            y.append(label_map[class_name])

            signer = fname.split("__")[1] if "__" in fname else ""
            manifest_rows.append({
                "index": len(X) - 1,
                "class": class_name,
                "signer": signer,
                "source_file": fpath,
                "n_frames_raw": raw.shape[0],
            })

    X = np.stack(X).astype(np.float32)          # (n_clips, SEQ_LEN, 225)
    mask_arr = np.stack(mask_list).astype(bool)  # (n_clips, SEQ_LEN)
    y = np.array(y, dtype=np.int64)              # (n_clips,)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "mask.npy"), mask_arr)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as fh:
        json.dump(label_map, fh, indent=2)

    with open(os.path.join(OUTPUT_DIR, "manifest.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["index", "class", "signer", "source_file", "n_frames_raw"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Compiled {X.shape[0]} clips -> X.npy {X.shape}, mask.npy {mask_arr.shape}, y.npy {y.shape}")
    print(f"Classes: {label_map}")
    n_clipped = sum(1 for r in manifest_rows if r["n_frames_raw"] > SEQ_LEN)
    n_padded = sum(1 for r in manifest_rows if r["n_frames_raw"] < SEQ_LEN)
    print(f"Clipped (raw > {SEQ_LEN}): {n_clipped}  |  Padded (raw < {SEQ_LEN}): {n_padded}  |  Exact: {len(manifest_rows) - n_clipped - n_padded}")
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
