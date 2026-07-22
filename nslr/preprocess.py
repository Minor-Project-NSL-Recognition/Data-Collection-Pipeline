"""Dual-anchor normalization + fixed-length standardization.

This is the shared preprocessing core (proposal 3.3-3.4) that must eventually be
mirrored in JavaScript for the browser. Normalize BEFORE padding: an undetected
block is all-zero and the anchor formulas map zero to zero, so padded frames stay
exactly zero and are trivially maskable.
"""

import json
import os

import numpy as np

from . import config as C


def normalize_hand_block(flat_block):
    """(h_i - h_0) / (||h_9 - h_0|| + eps) on one hand's flat (63,) block."""
    pts = flat_block.reshape(C.HAND_LANDMARKS, 3)
    wrist = pts[C.HAND_WRIST]
    scale = np.linalg.norm(pts[C.HAND_MIDDLE_FINGER_MCP] - wrist) + C.EPS
    return ((pts - wrist) / scale).reshape(-1)


def normalize_pose_block(flat_block):
    """(p_i - mid_shoulders) / (shoulder_width + eps) on the flat (99,) pose block."""
    pts = flat_block.reshape(C.POSE_LANDMARKS, 3)
    mid = 0.5 * (pts[C.POSE_LEFT_SHOULDER] + pts[C.POSE_RIGHT_SHOULDER])
    scale = np.linalg.norm(pts[C.POSE_LEFT_SHOULDER] - pts[C.POSE_RIGHT_SHOULDER]) + C.EPS
    return ((pts - mid) / scale).reshape(-1)


def normalize_clip(clip):
    """Normalize every frame of a (n_frames, 225) clip, blocks independent."""
    out = np.empty_like(clip)
    for t in range(clip.shape[0]):
        out[t, C.POSE_SLICE] = normalize_pose_block(clip[t, C.POSE_SLICE])
        out[t, C.LEFT_HAND_SLICE] = normalize_hand_block(clip[t, C.LEFT_HAND_SLICE])
        out[t, C.RIGHT_HAND_SLICE] = normalize_hand_block(clip[t, C.RIGHT_HAND_SLICE])
    return out


def standardize_length(clip, seq_len):
    """Return (fixed, mask, mode). Over-length -> uniform temporal subsampling
    (spans the whole clip); under-length -> zero-pad at the end."""
    n = clip.shape[0]
    if n == seq_len:
        return clip.astype(np.float32), np.ones(seq_len, dtype=bool), "exact"
    if n > seq_len:
        idx = np.linspace(0, n - 1, seq_len).round().astype(int)
        return clip[idx].astype(np.float32), np.ones(seq_len, dtype=bool), "subsampled"
    fixed = np.zeros((seq_len, C.FEATURE_DIM), dtype=np.float32)
    fixed[:n] = clip
    mask = np.zeros(seq_len, dtype=bool)
    mask[:n] = True
    return fixed, mask, "padded"


def load_seq_len(processed_dir):
    """SEQ_LEN from <processed>/seq_len.json, or FALLBACK_SEQ_LEN if absent."""
    path = os.path.join(processed_dir, "seq_len.json")
    try:
        with open(path) as fh:
            return int(json.load(fh)["seq_len"])
    except (FileNotFoundError, KeyError, ValueError, TypeError, json.JSONDecodeError, OSError):
        return C.FALLBACK_SEQ_LEN
