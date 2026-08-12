"""Live webcam tester for the trained model — perform a sign, see the prediction.

Segment-based (matches how clips were recorded and how the model was trained):
press SPACE to start a sign, SPACE again to stop and classify. This is more
reliable than continuous prediction because the model expects one whole sign per
input, and there is no "idle/nothing" class.

Controls:  SPACE = record/stop+classify  |  R = reset  |  L = landmarks
           F = fullscreen  |  Q / Esc = quit

The window itself is drawn by scripts/demo_ui.py — camera panel on the left,
result sidebar on the right — so it is presentable in front of an audience.

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
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)               # demo_ui.py lives next to this script

import cv2
import mediapipe as mp
import numpy as np

import demo_ui
from nslr import config as C
from nslr import ood
from nslr.landmarks import extract_frame_vector
from nslr.preprocess import normalize_clip, standardize_length
from nslr.voice import Voice


WINDOW = "NSL Recognition"


def parse_size(text, flag):
    """'1280x720' -> (1280, 720)."""
    try:
        w, h = (int(v) for v in str(text).lower().split("x"))
        if w <= 0 or h <= 0:
            raise ValueError
        return w, h
    except ValueError:
        raise SystemExit(f"{flag} expects WIDTHxHEIGHT, e.g. 1280x720 (got {text!r})")


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


def make_result(predictor, ood_stats, buffer, seq_len, class_names, threshold):
    """Classify `buffer` and package the answer the way the sidebar renders it.

    `state` drives every colour and heading in the UI, so the decision order —
    open-set gate first, then the negative class, then the softmax threshold —
    lives here and nowhere else.

    `key` is the class key and is what audio playback keys off. It is carried
    separately from `phrase` precisely so playback never has to reverse a
    human-readable string back into a class."""
    if len(buffer) < 5:
        return {"state": "short", "phrase": "Clip too short", "category": "",
                "note": f"{len(buffer)} frames captured - need at least 5",
                "key": None, "conf": None, "top3": [], "dist": None,
                "t": time.time()}

    key_name, conf, top3, dist = classify(predictor, ood_stats, buffer, seq_len, class_names)
    phrase, category = demo_ui.split_label(key_name, C.CLASSES.get(key_name, key_name))
    common = {"key": key_name, "top3": top3, "dist": dist, "conf": conf,
              "category": category, "t": time.time()}

    if ood_stats is not None and dist > ood_stats[2]:
        return {"state": "ood", "phrase": "Unknown sign",
                "note": "does not match any of the known phrases", **common}
    # `none` is a real trained class (rest, random motion, partial signs), but it
    # is the model declining to answer, not an answer. Reporting it as a MATCH
    # put "No known sign" in the accepted slot at high confidence — and now that
    # the demo speaks, it would also have announced a phrase at someone who did
    # not sign one. The Dart offline path has always folded it in here; this is
    # the same rule. server/inference.py still needs it.
    if key_name == "none":
        return {"state": "ood", "phrase": "No known sign",
                "note": "matched the negative class", **common}
    if conf < threshold:
        return {"state": "low", "phrase": phrase,
                "note": f"below the {threshold:.0%} confidence threshold", **common}
    return {"state": "ok", "phrase": phrase,
            "note": category or "recognised", **common}


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
    p.add_argument("--mute", action="store_true",
                   help="do not play the Nepali audio for a recognised sign")
    p.add_argument("--cam-res", default="1280x720", metavar="WxH",
                   help="capture resolution to request from the camera (default: "
                        "1280x720). Landmark extraction costs more at higher "
                        "resolutions and short clips are zero-padded rather than "
                        "resampled, so if the fps readout goes amber, drop to "
                        "640x480. Pass 'auto' to take whatever the camera gives.")
    p.add_argument("--window-size", default=None, metavar="WxH",
                   help="initial window size, e.g. 1600x900 (default: fit the "
                        "composed layout). The window is drag-resizable either way.")
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
    want = None if a.cam_res in (None, "auto") else parse_size(a.cam_res, "--cam-res")
    if want:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, want[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want[1])
    ok, probe = cap.read()          # cameras only report the size they settled on after a read
    if not ok:
        raise SystemExit(f"Camera {a.cam} opened but returned no frames.")
    frame_h, frame_w = probe.shape[:2]
    if want and want != (frame_w, frame_h):
        print(f"Camera refused {want[0]}x{want[1]}; using {frame_w}x{frame_h}.")

    # Nepali is spoken, not drawn: PIL in this venv has no raqm, so it cannot
    # shape Devanagari and would place matras and conjuncts wrongly. The sidebar
    # stays English; the phone app shows the script properly.
    voice = Voice(class_names, verbose=not a.mute)
    if a.mute:
        voice.enabled = False

    labels = {k: demo_ui.split_label(k, C.CLASSES.get(k, k)) for k in class_names}
    ui = demo_ui.Dashboard(frame_w, frame_h, labels=labels, seq_len=seq_len,
                           threshold=threshold, cam_index=a.cam)
    if not ui.type.truetype:
        print("No TrueType font found (or Pillow missing) — falling back to the "
              "OpenCV font. Layout is unchanged.")

    # WINDOW_NORMAL, not imshow's implicit WINDOW_AUTOSIZE, is what makes it
    # resizable; KEEPRATIO stops a careless drag from stretching the video.
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    win = parse_size(a.window_size, "--window-size") if a.window_size else ui.window_size()
    cv2.resizeWindow(WINDOW, win[0], win[1])
    fullscreen = False

    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    drawing, styles = mp.solutions.drawing_utils, mp.solutions.drawing_styles
    mp_h = mp.solutions.holistic

    recording, buffer, rec_start = False, [], 0.0
    result, show_landmarks = None, True
    fps, last = 0.0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)                       # same mirror as record.py
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        if show_landmarks:
            drawing.draw_landmarks(frame, results.pose_landmarks, mp_h.POSE_CONNECTIONS,
                                   landmark_drawing_spec=styles.get_default_pose_landmarks_style())
            for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
                drawing.draw_landmarks(frame, hand, mp_h.HAND_CONNECTIONS,
                                       landmark_drawing_spec=styles.get_default_hand_landmarks_style())

        if recording:
            vec, _ = extract_frame_vector(results)
            buffer.append(vec)

        now = time.time()
        dt = now - last
        last = now
        if dt > 0:                                       # smoothed, or it flickers
            fps = 1.0 / dt if fps == 0 else fps * 0.9 + 0.1 / dt

        cv2.imshow(WINDOW, ui.render(
            frame, recording=recording, rec_frames=len(buffer),
            rec_time=now - rec_start if recording else 0.0,
            result=result, fps=fps, now=now))
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if not recording:
                recording, buffer, rec_start = True, [], time.time()
                result = None
            else:
                recording = False
                result = make_result(predictor, ood_stats, buffer, seq_len,
                                     class_names, threshold)
                # Speak ONLY on a match. `low` and `ood` are the model
                # declining to commit, and announcing a phrase it just rejected
                # is worse than saying nothing at all.
                if result["state"] == "ok":
                    voice.play(result["key"])
        elif key in (ord("r"), ord("R")):
            result = None
        elif key in (ord("l"), ord("L")):
            show_landmarks = not show_landmarks
        elif key in (ord("f"), ord("F")):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif key in (ord("q"), ord("Q"), 27):
            break

    voice.stop()
    cap.release()
    holistic.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
