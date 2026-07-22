"""Live webcam tester for the trained model — perform a sign, see the prediction.

Segment-based (matches how clips were recorded and how the model was trained):
press SPACE to start a sign, SPACE again to stop and classify. This is more
reliable than continuous prediction because the model expects one whole sign per
input, and there is no "idle/nothing" class.

Controls:  SPACE = record/stop+classify   |   R = clear result   |   Q / Esc = quit

Needs a model from train_model.py. Run:
    python scripts/train_model.py
    python scripts/live_demo.py
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np

from nslr import config as C
from nslr.landmarks import extract_frame_vector
from nslr.preprocess import normalize_clip, standardize_length


def draw_panel(frame, lines, org=(10, 10), pad=8):
    """Draw a translucent black box with white text lines at the top-left."""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t, _ in lines]
    w = max((s[0] for s in sizes), default=0) + 2 * pad
    h = sum(s[1] + 10 for s in sizes) + pad
    x, y = org
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cy = y + pad + 12
    for (text, color) in lines:
        cv2.putText(frame, text, (x + pad, cy), font, scale, color, thick, cv2.LINE_AA)
        cy += cv2.getTextSize(text, font, scale, thick)[0][1] + 10


def classify(model, buffer, seq_len, class_names, threshold):
    """buffer: list of raw 225-vectors -> (label_key, confidence, top3)."""
    clip = np.stack(buffer).astype(np.float32)
    fixed, _, _ = standardize_length(normalize_clip(clip), seq_len)
    probs = model.predict(fixed[None, ...], verbose=0)[0]
    order = np.argsort(probs)[::-1]
    top3 = [(class_names[i], float(probs[i])) for i in order[:3]]
    best_i = int(order[0])
    return class_names[best_i], float(probs[best_i]), top3


def main():
    p = argparse.ArgumentParser(description="Live webcam tester for the NSL model.")
    p.add_argument("--results", default=C.RESULTS_DIR)
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--threshold", type=float, default=None,
                   help="override the confidence threshold stored in model_meta.json")
    a = p.parse_args()

    model_path = os.path.join(a.results, "model.keras")
    meta_path = os.path.join(a.results, "model_meta.json")
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        raise SystemExit(f"No model in {a.results}. Run:  python scripts/train_model.py")

    from tensorflow import keras
    model = keras.models.load_model(model_path)
    with open(meta_path) as fh:
        meta = json.load(fh)
    class_names = meta["class_names"]
    seq_len = meta["seq_len"]
    threshold = a.threshold if a.threshold is not None else meta.get("confidence_threshold", 0.75)

    cap = cv2.VideoCapture(a.cam)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {a.cam}")
    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    drawing, styles = mp.solutions.drawing_utils, mp.solutions.drawing_styles
    mp_h = mp.solutions.holistic

    recording, buffer, rec_start = False, [], 0.0
    result_line = ("Press SPACE to record a sign", (200, 200, 200))
    top3_lines = []

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)                       # same mirror as record.py
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        drawing.draw_landmarks(frame, results.pose_landmarks, mp_h.POSE_CONNECTIONS,
                               landmark_drawing_spec=styles.get_default_pose_landmarks_style())
        for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
            drawing.draw_landmarks(frame, hand, mp_h.HAND_CONNECTIONS,
                                   landmark_drawing_spec=styles.get_default_hand_landmarks_style())

        if recording:
            vec, _ = extract_frame_vector(results)
            buffer.append(vec)

        panel = [("SPACE record/stop   R clear   Q quit", (180, 180, 180))]
        if recording:
            panel.append((f"REC  {len(buffer)} frames  {time.time() - rec_start:0.1f}s", (0, 0, 255)))
        panel.append(result_line)
        panel.extend(top3_lines)
        draw_panel(frame, panel)

        cv2.imshow("NSL live tester", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if not recording:
                recording, buffer, rec_start = True, [], time.time()
                result_line = ("Recording... SPACE to stop", (0, 220, 220))
                top3_lines = []
            else:
                recording = False
                if len(buffer) < 5:
                    result_line = (f"Too short ({len(buffer)} frames)", (0, 165, 255))
                    top3_lines = []
                else:
                    key_name, conf, top3 = classify(model, buffer, seq_len, class_names, threshold)
                    label = C.CLASSES.get(key_name, key_name)
                    if conf >= threshold:
                        result_line = (f"{label}   {conf:.0%}", (0, 255, 0))
                    else:
                        result_line = (f"Not confident (best: {key_name} {conf:.0%})", (0, 165, 255))
                    top3_lines = [(f"   {k}: {c:.0%}", (200, 200, 200)) for k, c in top3]
        elif key in (ord("r"), ord("R")):
            result_line = ("Press SPACE to record a sign", (200, 200, 200))
            top3_lines = []
        elif key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
