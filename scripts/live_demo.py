"""Live webcam tester for the trained model — perform a sign, see the prediction.

Segment-based (matches how clips were recorded and how the model was trained):
press SPACE to start a sign, SPACE again to stop and classify. This is more
reliable than continuous prediction because the model expects one whole sign per
input, and there is no "idle/nothing" class.

Controls:  SPACE = record/stop+classify   |   R = clear result   |   Q / Esc = quit

Needs a model from train_model.py. Run:
    python scripts/train_model.py
    python scripts/live_demo.py
    python scripts/live_demo.py --cam-res 1280x720 --window-size 1600x900
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
from nslr import ood
from nslr.landmarks import extract_frame_vector
from nslr.preprocess import normalize_clip, standardize_length


WINDOW = "NSL live tester"


def parse_size(text, flag):
    """'1280x720' -> (1280, 720)."""
    try:
        w, h = (int(v) for v in str(text).lower().split("x"))
        if w <= 0 or h <= 0:
            raise ValueError
        return w, h
    except ValueError:
        raise SystemExit(f"{flag} expects WIDTHxHEIGHT, e.g. 1280x720 (got {text!r})")


def draw_panel(frame, lines, org=(10, 10), scale=0.6):
    """Draw a translucent black box with white text lines at the top-left.
    Everything is derived from `scale` so the panel keeps its proportions at any
    capture resolution (scale=0.6 reproduces the original 640-wide layout)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = max(1, round(scale * 1.6))
    pad, gap = round(scale * 13), round(scale * 17)
    sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t, _ in lines]
    w = max((s[0] for s in sizes), default=0) + 2 * pad
    h = sum(s[1] + gap for s in sizes) + pad
    x, y = org
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cy = y + pad + round(scale * 20)
    for (text, color) in lines:
        cv2.putText(frame, text, (x + pad, cy), font, scale, color, thick, cv2.LINE_AA)
        cy += cv2.getTextSize(text, font, scale, thick)[0][1] + gap


def classify(predictor, ood_stats, buffer, seq_len, class_names):
    """buffer: list of raw 225-vectors -> (label_key, confidence, top3, ood_distance).
    predictor outputs [softmax, embedding]; ood_distance is None if no gate is loaded."""
    clip = np.stack(buffer).astype(np.float32)
    fixed, _, _ = standardize_length(normalize_clip(clip), seq_len)
    probs, z = predictor.predict(fixed[None, ...], verbose=0)
    probs = probs[0]
    order = np.argsort(probs)[::-1]
    top3 = [(class_names[i], float(probs[i])) for i in order[:3]]
    dist = None
    if ood_stats is not None:
        means, precision, _ = ood_stats
        dist = float(ood.mahalanobis_min(z, means, precision)[0][0])
    return class_names[int(order[0])], float(probs[order[0]]), top3, dist


def main():
    p = argparse.ArgumentParser(description="Live webcam tester for the NSL model.")
    p.add_argument("--results", default=C.RESULTS_DIR)
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--threshold", type=float, default=None,
                   help="override the softmax confidence threshold stored in model_meta.json")
    p.add_argument("--ood-threshold", type=float, default=None,
                   help="override the open-set reject distance (higher = more lenient)")
    p.add_argument("--no-ood", action="store_true",
                   help="disable open-set rejection entirely (accept whatever softmax picks)")
    p.add_argument("--cam-res", default=None, metavar="WxH",
                   help="capture resolution to request from the camera, e.g. 1280x720 "
                        "(default: whatever the camera gives)")
    p.add_argument("--window-size", default=None, metavar="WxH",
                   help="initial window size, e.g. 1600x900 (default: match the frame). "
                        "The window is drag-resizable either way.")
    a = p.parse_args()

    model_path = os.path.join(a.results, "model.keras")
    meta_path = os.path.join(a.results, "model_meta.json")
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        raise SystemExit(f"No model in {a.results}. Run:  python scripts/train_model.py")

    from tensorflow import keras
    model = keras.models.load_model(model_path)
    # one forward pass -> [softmax, embedding], so the gate reuses the same features.
    predictor = keras.Model(model.inputs, [model.layers[-1].output, model.layers[-2].output])
    with open(meta_path) as fh:
        meta = json.load(fh)
    class_names = meta["class_names"]
    seq_len = meta["seq_len"]
    threshold = a.threshold if a.threshold is not None else meta.get("confidence_threshold", 0.75)

    ood_stats = None
    ood_path = os.path.join(a.results, "ood_stats.npz")
    if a.no_ood:
        print("Open-set rejection DISABLED (--no-ood): showing softmax's top pick.")
    elif os.path.exists(ood_path):
        means, precision, ood_thr, _ = ood.load_stats(ood_path)
        if a.ood_threshold is not None:
            ood_thr = a.ood_threshold
        ood_stats = (means, precision, ood_thr)
        print(f"Open-set gate ON (reject distance > {ood_thr:.1f}). "
              f"Watch the [dist ..] readout; raise with --ood-threshold or turn off with --no-ood.")
    else:
        print("No ood_stats.npz — running without open-set rejection.")

    cap = cv2.VideoCapture(a.cam)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {a.cam}")
    want = parse_size(a.cam_res, "--cam-res") if a.cam_res else None
    if want:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, want[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want[1])
    ok, probe = cap.read()          # cameras only report the size they settled on after a read
    if not ok:
        raise SystemExit(f"Camera {a.cam} opened but returned no frames.")
    frame_h, frame_w = probe.shape[:2]
    if want and want != (frame_w, frame_h):
        print(f"Camera refused {want[0]}x{want[1]}; using {frame_w}x{frame_h}.")
    hud_scale = 0.6 * frame_w / 640          # keep the overlay proportional at any resolution

    # WINDOW_NORMAL, not imshow's implicit WINDOW_AUTOSIZE, is what makes it resizable.
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    win = parse_size(a.window_size, "--window-size") if a.window_size else (frame_w, frame_h)
    cv2.resizeWindow(WINDOW, win[0], win[1])

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
        draw_panel(frame, panel, scale=hud_scale)

        cv2.imshow(WINDOW, frame)
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
                    key_name, conf, top3, dist = classify(predictor, ood_stats, buffer, seq_len, class_names)
                    label = C.CLASSES.get(key_name, key_name)
                    is_ood = ood_stats is not None and dist > ood_stats[2]
                    if is_ood:
                        result_line = (f"Unknown sign — rejected (dist {dist:.1f})", (0, 0, 255))
                    elif conf >= threshold:
                        result_line = (f"{label}   {conf:.0%}", (0, 255, 0))
                    else:
                        result_line = (f"Not confident (best: {key_name} {conf:.0%})", (0, 165, 255))
                    dtxt = f"  [dist {dist:.1f}]" if dist is not None else ""
                    top3_lines = [(f"   {k}: {c:.0%}", (200, 200, 200)) for k, c in top3]
                    if dtxt:
                        top3_lines.append((dtxt.strip(), (150, 150, 150)))
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
