"""Replay a clip captured ON THE PHONE through the Python pipeline, and diff it
against the training distribution.

This answers the question "same model, same sign, why does the phone get it wrong?"
by removing every variable except the landmarks themselves. The app dumps the raw
225-vectors it built; this feeds those exact numbers to model.keras via the same
normalize -> standardize path live_demo.py uses.

    adb pull /sdcard/Android/data/np.edu.nsl.nsl_app/files/clips
    python scripts/inspect_device_clip.py clips/clip_2026-08-11T15-12-03.f32

How to read the result
----------------------
* Python AGREES with the phone (same wrong answer, similar distance)
      -> the landmarks are the problem. The comparison table below says which
         part: detection coverage, coordinate range, or geometry.
* Python gets it RIGHT from the phone's own landmarks
      -> the landmarks are fine and the Dart/TFLite inference path is at fault.
         (Unlikely -- the golden tests pin preprocessing and export_tflite
         verifies the model to 1e-7 -- but this is what would prove it.)

Every statistic is printed beside the same statistic over data/raw, because an
absolute number means nothing here; only the deviation does.
"""

import argparse
import glob
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nslr import config as C
from nslr import ood
from nslr.preprocess import normalize_clip, standardize_length

BLOCKS = [("pose", C.POSE_SLICE, C.POSE_LANDMARKS),
          ("left_hand", C.LEFT_HAND_SLICE, C.HAND_LANDMARKS),
          ("right_hand", C.RIGHT_HAND_SLICE, C.HAND_LANDMARKS)]


def load_f32(path):
    """The app's dump format: little-endian float32, row-major (n, 225), no header."""
    raw = np.frombuffer(open(path, "rb").read(), dtype="<f4")
    if raw.size % C.FEATURE_DIM != 0:
        raise SystemExit(f"{path}: {raw.size} floats is not a multiple of {C.FEATURE_DIM}")
    return raw.reshape(-1, C.FEATURE_DIM).astype(np.float32)


def stats(clip):
    """Shape-and-coverage summary of one raw clip."""
    out = {"n_frames": int(len(clip))}
    for name, sl, npts in BLOCKS:
        block = clip[:, sl]
        on = np.abs(block).sum(axis=1) > 0
        out[f"{name}_detect"] = float(on.mean())
        if on.any():
            pts = block[on].reshape(-1, npts, 3)
            out[f"{name}_x"] = (float(pts[..., 0].min()), float(pts[..., 0].max()))
            out[f"{name}_y"] = (float(pts[..., 1].min()), float(pts[..., 1].max()))
            out[f"{name}_z_std"] = float(pts[..., 2].std())
    pose_on = np.abs(clip[:, C.POSE_SLICE]).sum(axis=1) > 0
    if pose_on.any():
        p = clip[pose_on][:, C.POSE_SLICE].reshape(-1, C.POSE_LANDMARKS, 3)
        ls, rs = p[:, C.POSE_LEFT_SHOULDER, :], p[:, C.POSE_RIGHT_SHOULDER, :]
        out["shoulder_width"] = float(np.linalg.norm(ls - rs, axis=1).mean())
        # Signed x gap: its SIGN encodes which shoulder is on which side of the
        # image, so a flipped sign means the frame's mirroring differs from
        # record.py's -- which silently swaps the two hand blocks.
        out["shoulder_dx_signed"] = float((ls[:, 0] - rs[:, 0]).mean())
    return out


def training_stats(raw_dir, class_names, limit_per_class=25):
    acc = []
    for name in class_names:
        cdir = os.path.join(raw_dir, name)
        if not os.path.isdir(cdir):
            continue
        files = sorted(f for f in os.listdir(cdir) if f.endswith(".npy"))[:limit_per_class]
        for f in files:
            a = np.load(os.path.join(cdir, f)).astype(np.float32)
            if a.ndim == 2 and a.shape[1] == C.FEATURE_DIM:
                acc.append(stats(a))
    return acc


def agg(rows, key):
    vals = [r[key] for r in rows if key in r]
    if not vals:
        return None
    if isinstance(vals[0], tuple):
        return (float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals])))
    return float(np.mean(vals))


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, tuple):
        return f"[{v[0]:+.3f}, {v[1]:+.3f}]"
    return f"{v:+.4f}"


def main():
    p = argparse.ArgumentParser(description="Diagnose a phone-captured landmark clip.")
    p.add_argument("clip", nargs="+", help=".f32 dump(s) pulled from the device")
    p.add_argument("--results", default=C.RESULTS_DIR)
    p.add_argument("--data", default=C.RAW_DIR)
    p.add_argument("--truth", default=None, help="the phrase you actually performed")
    a = p.parse_args()

    from tensorflow import keras

    meta = json.load(open(os.path.join(a.results, "model_meta.json")))
    class_names = meta["class_names"]
    seq_len = int(meta["seq_len"])
    conf_thr = float(meta.get("confidence_threshold", 0.75))
    gate = json.load(open(os.path.join(a.results, "ood.json")))
    means, precision = np.array(gate["means"]), np.array(gate["precision"])
    thr = float(gate["threshold"])

    model = keras.models.load_model(os.path.join(a.results, "model.keras"))
    predictor = keras.Model(model.inputs, [model.layers[-1].output, model.layers[-2].output])

    paths = []
    for pattern in a.clip:
        hits = glob.glob(pattern)
        paths.extend(hits if hits else [pattern])
    if not paths:
        raise SystemExit("no clips matched")

    print(f"\nreference: {len(class_names)} classes, seq_len {seq_len}, "
          f"conf>={conf_thr}, reject d>{thr:.2f}")
    print("building training reference statistics...")
    tr = training_stats(a.data, class_names)
    print(f"  from {len(tr)} training clips\n")

    for path in sorted(paths):
        clip = load_f32(path)
        s = stats(clip)
        fixed, _, mode = standardize_length(normalize_clip(clip.copy()), seq_len)
        probs, emb = predictor.predict(fixed[None], verbose=0)
        probs, emb = probs[0], emb[0]
        dist = float(ood.mahalanobis_min(emb[None], means, precision)[0][0])
        order = np.argsort(probs)[::-1]

        print("=" * 78)
        print(f"  {os.path.basename(path)}")
        print("=" * 78)
        print(f"  PYTHON says: {class_names[order[0]]} "
              f"{probs[order[0]]:.1%}   distance {dist:.2f}   ({mode})")
        print("    " + "  ".join(f"{class_names[i]} {probs[i]:.0%}" for i in order[:3]))
        if a.truth:
            ok = class_names[order[0]] == a.truth
            print(f"    truth={a.truth} -> {'CORRECT' if ok else 'WRONG'}")
        gates = []
        if probs[order[0]] < conf_thr:
            gates.append(f"below confidence {conf_thr}")
        if dist > thr:
            gates.append(f"beyond distance {thr:.2f}")
        print(f"    gates: {', '.join(gates) if gates else 'passes both'}")

        print(f"\n  {'statistic':22s} {'THIS CLIP':>20s} {'training mean':>20s}")
        print("  " + "-" * 64)
        keys = ["n_frames", "pose_detect", "left_hand_detect", "right_hand_detect",
                "shoulder_width", "shoulder_dx_signed",
                "pose_x", "pose_y", "pose_z_std",
                "left_hand_x", "left_hand_y", "right_hand_x", "right_hand_y"]
        for k in keys:
            mine = s.get(k)
            ref = agg(tr, k)
            flag = ""
            if isinstance(mine, float) and isinstance(ref, float) and ref not in (None, 0):
                if k == "shoulder_dx_signed" and np.sign(mine) != np.sign(ref):
                    flag = "  <== SIGN FLIPPED (mirroring differs; hand blocks swap)"
                elif abs(mine - ref) > 0.5 * abs(ref):
                    flag = "  <== differs >50%"
            print(f"  {k:22s} {fmt(mine):>20s} {fmt(ref):>20s}{flag}")
        print()

    print("If PYTHON reproduces the phone's answer, the landmarks are the problem and")
    print("the flagged rows above say which part. If PYTHON gets it right, the fault is")
    print("in the Dart/TFLite path instead.\n")


if __name__ == "__main__":
    main()
