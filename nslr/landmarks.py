"""MediaPipe Holistic result -> 225-vector [pose 99 | left hand 63 | right hand 63]."""

import numpy as np

from . import config as C


def extract_frame_vector(results):
    """Return (vector, flags): a 225-length float32 vector with missing blocks
    zero-filled, and a dict recording which components were detected."""
    pose = np.zeros(C.POSE_DIM, dtype=np.float32)
    left = np.zeros(C.HAND_DIM, dtype=np.float32)
    right = np.zeros(C.HAND_DIM, dtype=np.float32)
    flags = {"pose": False, "left_hand": False, "right_hand": False}

    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
                        dtype=np.float32).flatten()
        flags["pose"] = True
    if results.left_hand_landmarks:
        left = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                        dtype=np.float32).flatten()
        flags["left_hand"] = True
    if results.right_hand_landmarks:
        right = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                         dtype=np.float32).flatten()
        flags["right_hand"] = True

    vector = np.concatenate([pose, left, right])
    assert vector.shape[0] == C.FEATURE_DIM, f"expected {C.FEATURE_DIM}, got {vector.shape[0]}"
    return vector, flags
