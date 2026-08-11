"""Find the y-scale correction that makes phone-captured clips classify correctly.

MediaPipe returns x = px/frame_width, y = px/frame_height. record.py captured
LANDSCAPE (webcam default 640x480); the phone feeds an upright PORTRAIT frame
(480x640). So the two disagree about how many normalized units a vertical
centimetre is worth, by

    (H_phone/W_phone) / (H_train/W_train) = (640/480) / (480/640) = 1.78

`normalize_pose_block` divides by shoulder width -- a single scalar, and one that is
mostly HORIZONTAL -- so it cancels the x half of that mismatch and leaves the y half
untouched. The skeleton the model receives from the phone is the right width and
half the height.

Measured on device clips: y-span/shoulder-width is 3.26 against training's 6.44.

This sweeps a y multiplier over real device dumps and reports what each does to the
prediction. If a factor near 1.78 restores the correct class, the aspect mismatch is
the bug and that factor is the fix.

    python scripts/inspect_device_clip.py clips/*.f32          # first, to see the problem
    python scripts/fix_aspect_sweep.py clips/*.f32 --truth need_ambulance
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

BLOCKS = [(C.POSE_SLICE, C.POSE_LANDMARKS),
          (C.LEFT_HAND_SLICE, C.HAND_LANDMARKS),
          (C.RIGHT_HAND_SLICE, C.HAND_LANDMARKS)]


def scale_y(clip, f):
    """Multiply the y component of every detected landmark by f.

    Applied to RAW MediaPipe-normalized coordinates, which is where the frame's
    aspect ratio enters. Undetected blocks stay exactly zero so masking and the
    zero-maps-to-zero property of the anchor formulas are preserved.
    """
    out = clip.copy()
    for sl, npts in BLOCKS:
        block = out[:, sl].reshape(len(out), npts, 3)
        live = np.abs(block).sum(axis=(1, 2)) > 0
        block[live, :, 1] *= f
        out[:, sl] = block.reshape(len(out), -1)
    return out


def geometry(clip, seq_len):
    """y-span and x-span measured in shoulder-width units, post-normalization.

    This is the scale-free shape statistic the model actually sees, so it is what
    should match between the phone and the training data.
    """
    fixed, _, _ = standardize_length(normalize_clip(clip.copy()), seq_len)
    pose = fixed[:, C.POSE_SLICE].reshape(len(fixed), C.POSE_LANDMARKS, 3)
    live = np.abs(pose).sum(axis=(1, 2)) > 0
    if not live.any():
        return float("nan"), float("nan")
    p = pose[live]
    return (float(p[..., 0].max() - p[..., 0].min()),
            float(p[..., 1].max() - p[..., 1].min()))


def main():
    p = argparse.ArgumentParser(description="Sweep a y-aspect correction over device clips.")
    p.add_argument("clip", nargs="+")
    p.add_argument("--results", default=C.RESULTS_DIR)
    p.add_argument("--data", default=C.RAW_DIR)
    p.add_argument("--truth", default=None)
    a = p.parse_args()

    from tensorflow import keras

    meta = json.load(open(os.path.join(a.results, "model_meta.json")))
    class_names = meta["class_names"]
    seq_len = int(meta["seq_len"])
    gate = json.load(open(os.path.join(a.results, "ood.json")))
    means, precision = np.array(gate["means"]), np.array(gate["precision"])

    model = keras.models.load_model(os.path.join(a.results, "model.keras"))
    predictor = keras.Model(model.inputs, [model.layers[-1].output, model.layers[-2].output])

    paths = []
    for pattern in a.clip:
        hits = sorted(glob.glob(pattern))
        paths.extend(hits if hits else [pattern])
    if not paths:
        raise SystemExit("no clips matched")

    # Training reference for the same scale-free statistic.
    ref_x, ref_y = [], []
    for name in class_names:
        cdir = os.path.join(a.data, name)
        if not os.path.isdir(cdir):
            continue
        for f in sorted(x for x in os.listdir(cdir) if x.endswith(".npy"))[:20]:
            arr = np.load(os.path.join(cdir, f)).astype(np.float32)
            if arr.ndim == 2 and arr.shape[1] == C.FEATURE_DIM:
                gx, gy = geometry(arr, seq_len)
                if not np.isnan(gy):
                    ref_x.append(gx)
                    ref_y.append(gy)
    tx, ty = float(np.mean(ref_x)), float(np.mean(ref_y))
    print(f"\ntraining reference (normalized, shoulder-width units):"
          f"  x-span {tx:.2f}   y-span {ty:.2f}")

    factors = [1.0, 1.2, 1.4, 1.6, 1.78, 2.0, 2.2, 2.5]

    for path in paths:
        raw = np.frombuffer(open(path, "rb").read(), dtype="<f4")
        clip = raw.reshape(-1, C.FEATURE_DIM).astype(np.float32)
        gx, gy = geometry(clip, seq_len)
        print("\n" + "=" * 76)
        print(f"  {os.path.basename(path)}   {len(clip)} frames")
        print(f"  uncorrected: x-span {gx:.2f} (train {tx:.2f})   "
              f"y-span {gy:.2f} (train {ty:.2f})  -> y is {ty/max(gy,1e-9):.2f}x too small")
        print("=" * 76)
        print(f"  {'y*f':>6s}  {'y-span':>7s}  {'prediction':18s} {'conf':>6s} {'dist':>7s}  {'verdict':8s}")
        for f in factors:
            c = scale_y(clip, f)
            _, gy2 = geometry(c, seq_len)
            fixed, _, _ = standardize_length(normalize_clip(c), seq_len)
            probs, emb = predictor.predict(fixed[None], verbose=0)
            probs, emb = probs[0], emb[0]
            dist = float(ood.mahalanobis_min(emb[None], means, precision)[0][0])
            best = int(probs.argmax())
            verdict = ""
            if a.truth:
                verdict = "CORRECT" if class_names[best] == a.truth else "wrong"
            print(f"  {f:6.2f}  {gy2:7.2f}  {class_names[best]:18s} "
                  f"{probs[best]:5.0%} {dist:7.2f}  {verdict:8s}")

    print("\nIf the correct class appears around f=1.78, the landscape/portrait aspect")
    print("mismatch is the bug. The fix belongs in the app (scale y once, where the")
    print("frame dimensions are known) -- not in nslr/, which must keep matching the")
    print("training data.\n")


if __name__ == "__main__":
    main()
