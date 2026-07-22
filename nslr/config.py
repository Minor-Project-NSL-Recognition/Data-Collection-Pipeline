"""Locked constants and paths shared across the whole pipeline.

This is the single reference the proposal (4.3.5) asks for: the 225-feature
layout, normalization anchors, and default seq_len live here and nowhere else,
so the Python training path and the future JS browser path can be checked
against one source of truth.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# Folder-safe class name -> GUI label. Phrase #6 is need_toilet by design
# (the proposal text still says "earthquake"; the data is the source of truth).
CLASSES = {
    "cant_breathe":     "1. I can't breathe (Medical)",
    "building_on_fire": "2. The building is on fire (Fire)",
    "call_police":      "3. Call the police (Crime)",
    "need_ambulance":   "4. I need an ambulance (Medical)",
    "help_danger":      "5. Help me / I am in danger (Generic)",
    "need_toilet":      "6. I need to go to the toilet (Basic need)",
}

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
POSE_DIM = POSE_LANDMARKS * 3           # 99
HAND_DIM = HAND_LANDMARKS * 3           # 63
FEATURE_DIM = POSE_DIM + HAND_DIM * 2   # 225

# Slices into the 225-vector, in record.py's concatenation order.
POSE_SLICE = slice(0, POSE_DIM)
LEFT_HAND_SLICE = slice(POSE_DIM, POSE_DIM + HAND_DIM)
RIGHT_HAND_SLICE = slice(POSE_DIM + HAND_DIM, FEATURE_DIM)

# Normalization anchors (MediaPipe landmark indices).
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
HAND_WRIST = 0
HAND_MIDDLE_FINGER_MCP = 9

EPS = 1e-6
FALLBACK_SEQ_LEN = 151   # used only if seq_len.json is missing
