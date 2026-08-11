"""Locked constants and paths shared across the whole pipeline.

This is the single reference the proposal (4.3.5) asks for: the 225-feature
layout, normalization anchors, and default seq_len live here and nowhere else,
so the Python training path and the future JS browser path can be checked
against one source of truth.
"""

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# Phrase wording is authored in ONE file, loaded by both languages, so the
# Python labels and the Dart ones cannot drift. It sits under app/assets/
# because Flutter cannot bundle an asset from above its own package root; see
# the note inside the file. Class KEYS are frozen (the model trained on them),
# but wording is free to change here alone.
PHRASES_PATH = os.path.join(REPO_ROOT, "app", "assets", "phrases.json")
with open(PHRASES_PATH, encoding="utf-8") as _fh:
    PHRASES = json.load(_fh)["phrases"]

# Folder-safe class name -> GUI label, derived rather than declared. Kept in the
# historic "N. text (Category)" packed form that demo_ui.split_label, the
# server's `display` field and record.py's radio buttons all already parse.
# Phrase #6 is need_toilet by design (the proposal text still says "earthquake";
# the data is the source of truth). The 7th, `none`, is the open-set negative
# class -- rest, random motion, partial/mixed gestures. The pipeline ignores it
# until it has clips, then trains it as a real class.
CLASSES = {
    key: f"{p['order']}. {p['en']} ({p['category']})" for key, p in PHRASES.items()
}

# Folder-safe class name -> Nepali phrase. `none` maps to None, never a string:
# a rejected sign has nothing to display and nothing to speak.
NEPALI = {key: p["ne"] for key, p in PHRASES.items()}

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

# Effective capture rate of the recorded dataset, measured from the per-clip
# metadata (n_frames / duration_sec over all 570 clips): median 15.7, range
# 10.8-21.2. record.py never locked a frame rate -- it ran as fast as Tkinter
# plus MediaPipe allowed -- so this is an observed property of the data, not a
# setting. It matters at inference: the model reads frame COUNT as duration, so
# a client capturing the same sign at 30 fps produces twice the frames and a
# temporal scale the model has never seen. Any real-time path must decimate to
# roughly this rate before extracting landmarks.
TRAIN_CAPTURE_FPS = 15.7
