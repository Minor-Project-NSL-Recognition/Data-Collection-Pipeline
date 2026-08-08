"""Does the mobile landmark stack produce vectors our Holistic-trained model accepts?

The Flutter app cannot use MediaPipe Holistic (no mobile build exists), so it must
rebuild the 225-vector from PoseLandmarker + HandLandmarker. If those landmarks sit
in a different distribution than Holistic's, the trained model is worthless on the
phone -- and because record.py saved only landmarks and never video, that failure
would mean re-recording all 570 clips. This script answers the question before any
Dart is written.

Both extractors run on the *same* frames, so the only variable is the landmark
source. Each capture saves a paired clip; parity_report.py aggregates them.

    python scripts/parity_spike.py --selftest      # no camera: check the plumbing
    python scripts/parity_spike.py                 # webcam, SPACE to record a sign

Controls:  SPACE record/stop  |  1-7 pick the phrase you're signing  |  R clear  |  Q quit
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np

from nslr import config as C
from nslr import ood
from nslr.landmarks import extract_frame_vector
from nslr.preprocess import normalize_clip, standardize_length
from nslr.tasks_landmarks import TasksExtractor

WINDOW = "NSL parity spike  (Holistic vs Tasks)"
MODELS_DIR = os.path.join(C.REPO_ROOT, "models", "tasks")
PAIRS_DIR = os.path.join(C.REPO_ROOT, "data", "parity")


def load_predictor(results_dir):
    """Return (predict_fn, class_names, seq_len, threshold, ood_stats)."""
    from tensorflow import keras

    model_path = os.path.join(results_dir, "model.keras")
    meta_path = os.path.join(results_dir, "model_meta.json")
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        raise SystemExit(f"No model in {results_dir}. Run:  python scripts/train_model.py")

    model = keras.models.load_model(model_path)
    predictor = keras.Model(model.inputs, [model.layers[-1].output, model.layers[-2].output])
    with open(meta_path) as fh:
        meta = json.load(fh)

    ood_stats = None
    ood_path = os.path.join(results_dir, "ood_stats.npz")
    if os.path.exists(ood_path):
        means, precision, thr, _ = ood.load_stats(ood_path)
        ood_stats = (means, precision, thr)

    return (predictor, meta["class_names"], meta["seq_len"],
            meta.get("confidence_threshold", 0.75), ood_stats)


def classify(predictor, ood_stats, clip, seq_len, class_names):
    """Raw (frames,225) clip -> dict with label, confidence, top3, ood distance."""
    fixed, _, mode = standardize_length(normalize_clip(clip.astype(np.float32)), seq_len)
    probs, z = predictor.predict(fixed[None, ...], verbose=0)
    probs = probs[0]
    order = np.argsort(probs)[::-1]
    dist = None
    if ood_stats is not None:
        means, precision, _ = ood_stats
        dist = float(ood.mahalanobis_min(z, means, precision)[0][0])
    return {"label": class_names[int(order[0])],
            "conf": float(probs[order[0]]),
            "top3": [(class_names[i], float(probs[i])) for i in order[:3]],
            "probs": probs.tolist(),
            "dist": dist,
            "standardize": mode}


def selftest(args):
    """Run both extractors headlessly on synthetic frames: shapes, dtypes, wiring."""
    print("Self-test: no camera, synthetic frames — checking plumbing only.\n")
    tasks = TasksExtractor(os.path.join(MODELS_DIR, "pose_landmarker_full.task"),
                           os.path.join(MODELS_DIR, "hand_landmarker.task"))
    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

    rng = np.random.default_rng(0)
    h_clip, t_clip = [], []
    for i in range(12):
        rgb = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        h_vec, _ = extract_frame_vector(holistic.process(rgb))
        t_vec, t_flags = tasks.extract(rgb, i * 60)
        h_clip.append(h_vec)
        t_clip.append(t_vec)

    h_clip, t_clip = np.stack(h_clip), np.stack(t_clip)
    print(f"  holistic clip {h_clip.shape} {h_clip.dtype}")
    print(f"  tasks    clip {t_clip.shape} {t_clip.dtype}")
    assert h_clip.shape == t_clip.shape == (12, C.FEATURE_DIM)

    predictor, class_names, seq_len, _, ood_stats = load_predictor(args.results)
    for name, clip in (("holistic", h_clip), ("tasks", t_clip)):
        r = classify(predictor, ood_stats, clip, seq_len, class_names)
        print(f"  {name:9s} -> {r['label']:17s} {r['conf']:.0%}  dist={r['dist']:.1f}")

    tasks.close()
    holistic.close()
    print("\nPlumbing OK — both sources feed the model. Now run without --selftest,")
    print("at your webcam, and record several clips per phrase.")


def main():
    p = argparse.ArgumentParser(description="Holistic vs MediaPipe-Tasks landmark parity.")
    p.add_argument("--results", default=C.RESULTS_DIR)
    p.add_argument("--models", default=MODELS_DIR)
    p.add_argument("--out", default=PAIRS_DIR, help="where paired clips are saved")
    p.add_argument("--signer", default="signer01")
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--selftest", action="store_true", help="headless plumbing check")
    a = p.parse_args()

    if a.selftest:
        return selftest(a)

    keys = list(C.CLASSES.keys())
    predictor, class_names, seq_len, threshold, ood_stats = load_predictor(a.results)
    print(f"classes {class_names} | seq_len {seq_len}")

    tasks = TasksExtractor(os.path.join(a.models, "pose_landmarker_full.task"),
                           os.path.join(a.models, "hand_landmarker.task"))
    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

    cap = cv2.VideoCapture(a.cam)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {a.cam}")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    truth_idx = 0
    recording, h_buf, t_buf, agree_flags = False, [], [], []
    rec_start, t0 = 0.0, time.time()
    lines = [("SPACE to record a sign", (200, 200, 200))]

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)                    # same mirror as record.py
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h_res = holistic.process(rgb)
        ts_ms = int((time.time() - t0) * 1000)
        t_vec, t_flags = tasks.extract(rgb, ts_ms)

        if recording:
            h_vec, _ = extract_frame_vector(h_res)
            h_buf.append(h_vec)
            t_buf.append(t_vec)
            if t_flags["label_agrees"] is not None:
                agree_flags.append(t_flags["label_agrees"])

        hud = [(f"truth [{truth_idx + 1}]: {keys[truth_idx]}", (0, 255, 255)),
               ("SPACE rec  1-7 phrase  R clear  Q quit", (170, 170, 170))]
        if recording:
            hud.append((f"REC {len(h_buf)} frames  {time.time() - rec_start:0.1f}s",
                        (0, 0, 255)))
        hud.extend(lines)

        y = 26
        for text, color in hud:
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 1, cv2.LINE_AA)
            y += 24

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if not recording:
                recording, h_buf, t_buf, agree_flags = True, [], [], []
                rec_start = time.time()
                lines = [("recording...", (0, 220, 220))]
            else:
                recording = False
                if len(h_buf) < 5:
                    lines = [(f"too short ({len(h_buf)} frames)", (0, 165, 255))]
                else:
                    lines = _finish(np.stack(h_buf), np.stack(t_buf), agree_flags,
                                    time.time() - rec_start, keys[truth_idx], a,
                                    predictor, ood_stats, seq_len, class_names, threshold)
        elif ord("1") <= key <= ord("7"):
            truth_idx = min(key - ord("1"), len(keys) - 1)
        elif key in (ord("r"), ord("R")):
            lines = [("SPACE to record a sign", (200, 200, 200))]
        elif key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    tasks.close()
    holistic.close()
    cv2.destroyAllWindows()
    print(f"\nPaired clips saved under {a.out}")
    print("Aggregate them with:  python scripts/parity_report.py")


def _finish(h_clip, t_clip, agree_flags, duration, truth, a,
            predictor, ood_stats, seq_len, class_names, threshold):
    """Classify both clips, save the pair, and return HUD lines."""
    h = classify(predictor, ood_stats, h_clip, seq_len, class_names)
    t = classify(predictor, ood_stats, t_clip, seq_len, class_names)

    folder = os.path.join(a.out, truth)
    os.makedirs(folder, exist_ok=True)
    idx = 1 + max([int(f.split("__")[-2]) for f in os.listdir(folder)
                   if f.endswith("__holistic.npy")] or [0])
    stem = f"{truth}__{a.signer}__{idx:03d}"
    np.save(os.path.join(folder, stem + "__holistic.npy"), h_clip)
    np.save(os.path.join(folder, stem + "__tasks.npy"), t_clip)
    with open(os.path.join(folder, stem + ".json"), "w") as fh:
        json.dump({"class": truth, "signer": a.signer, "n_frames": int(len(h_clip)),
                   "duration_sec": round(duration, 2),
                   "fps": round(len(h_clip) / max(duration, 1e-6), 1),
                   "handedness_label_agreement": (
                       round(float(np.mean(agree_flags)), 3) if agree_flags else None),
                   "recorded_at": datetime.now().isoformat(timespec="seconds"),
                   "holistic": {k: h[k] for k in ("label", "conf", "dist")},
                   "tasks": {k: t[k] for k in ("label", "conf", "dist")}}, fh, indent=2)

    def col(r):
        return (0, 255, 0) if (r["label"] == truth and r["conf"] >= threshold) else (0, 165, 255)

    fps = len(h_clip) / max(duration, 1e-6)
    print(f"[{stem}] {len(h_clip)}f @ {fps:.1f}fps | "
          f"holistic {h['label']} {h['conf']:.0%} d={h['dist']:.1f} | "
          f"tasks {t['label']} {t['conf']:.0%} d={t['dist']:.1f}")
    return [(f"holistic: {h['label']} {h['conf']:.0%}  d={h['dist']:.1f}", col(h)),
            (f"tasks:    {t['label']} {t['conf']:.0%}  d={t['dist']:.1f}", col(t)),
            (f"saved {stem}  ({fps:.1f} fps)", (170, 170, 170))]


if __name__ == "__main__":
    main()
