"""Choose the open-set distance threshold from the DEPLOYMENT distribution.

Why the shipped threshold rejects real signs
--------------------------------------------
`train_model.py` fits the threshold as the p99 of *training* distances: same
signers, a webcam, and MediaPipe **Holistic** landmarks. The phone is none of
those -- it runs **MediaPipe Tasks** (Pose + Hand landmarkers), because Holistic
has no mobile build. If Tasks embeddings sit further from the class prototypes
than Holistic ones do, every real sign on the phone lands outside a threshold
fitted without a single Tasks frame in it, and the app answers `unknown`.

That is measurable rather than assumed, because `data/parity/` holds BOTH
extractions of the SAME recorded frames:

    <class>__<signer>__<n>__holistic.npy    control
    <class>__<signer>__<n>__tasks.npy       what the phone actually feeds the model

Comparing those two isolates the landmark source from framing, signer and
session -- they are the same frames. If tasks distances are much larger, the
threshold simply needs refitting on them. If BOTH are large, the phone pipeline
is not the problem and re-thresholding would be papering over a real bug.

What it prints
--------------
A sweep over candidate thresholds with, at each one:
  * accept rate on real Tasks clips   (want high  -- these are genuine signs)
  * accept rate on real Holistic clips
  * reject rate on synthetic OOD      (want high  -- these are not signs)
Then a recommendation at a target true-accept rate.

    python scripts/calibrate_ood.py
    python scripts/calibrate_ood.py --target 0.99
    python scripts/calibrate_ood.py --target 0.99 --apply     # write it back

`--apply` updates `ood_threshold` in results/model_meta.json and `threshold` in
results/ood.json. It does NOT retrain and does NOT touch the prototypes, the
model, or model.tflite -- only the single scalar the gate compares against.
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


def load_gate(results_dir):
    with open(os.path.join(results_dir, "ood.json")) as fh:
        gate = json.load(fh)
    with open(os.path.join(results_dir, "model_meta.json")) as fh:
        meta = json.load(fh)
    return (np.array(gate["means"]), np.array(gate["precision"]),
            float(gate["threshold"]), meta)


def batch(clips, seq_len):
    """Raw clips -> (n, seq_len, 225) model input."""
    out = np.empty((len(clips), seq_len, C.FEATURE_DIM), dtype=np.float32)
    for i, clip in enumerate(clips):
        fixed, _, _ = standardize_length(normalize_clip(clip.astype(np.float32)), seq_len)
        out[i] = fixed
    return out


def load_raw(raw_dir, class_names):
    clips, labels = [], []
    for ci, name in enumerate(class_names):
        cdir = os.path.join(raw_dir, name)
        if not os.path.isdir(cdir):
            continue
        for f in sorted(x for x in os.listdir(cdir) if x.endswith(".npy")):
            a = np.load(os.path.join(cdir, f)).astype(np.float32)
            if a.ndim == 2 and a.shape[1] == C.FEATURE_DIM:
                clips.append(a)
                labels.append(ci)
    return clips, np.array(labels)


def load_parity(pairs_dir, class_names, kind):
    """kind is 'tasks' or 'holistic'."""
    clips, labels = [], []
    for path in sorted(glob.glob(os.path.join(pairs_dir, "*", f"*__{kind}.npy"))):
        cls = os.path.basename(os.path.dirname(path))
        if cls not in class_names:
            continue
        a = np.load(path).astype(np.float32)
        if a.ndim == 2 and a.shape[1] == C.FEATURE_DIM and len(a) >= 5:
            clips.append(a)
            labels.append(class_names.index(cls))
    return clips, np.array(labels)


def synth_ood(X, y, n=300, seed=42):
    """Not-a-sign inputs built from real ones: half time-shuffled (motion order
    destroyed), half cross-class mixes. Same construction as train_model.py, so
    the reject rates here are comparable to what that script reported."""
    rng = np.random.default_rng(seed)
    half = X.shape[1] // 2
    out = []
    for _ in range(n // 2):
        c = X[rng.integers(len(X))].copy()
        rng.shuffle(c)
        out.append(c)
    for _ in range(n - n // 2):
        i = rng.integers(len(X))
        j = rng.integers(len(X))
        while y[j] == y[i]:
            j = rng.integers(len(X))
        out.append(np.concatenate([X[i][:half], X[j][half:]], axis=0))
    return np.stack(out).astype(np.float32)


def describe(name, dist, thr):
    if len(dist) == 0:
        print(f"  {name:34s} (none available)")
        return
    print(f"  {name:34s} n={len(dist):4d}  "
          f"p50={np.median(dist):6.2f}  p95={np.percentile(dist,95):6.2f}  "
          f"max={dist.max():6.2f}  beyond {thr:.2f}: {100*(dist>thr).mean():5.1f}%")


def main():
    p = argparse.ArgumentParser(description="Refit the open-set threshold on deployment data.")
    p.add_argument("--results", default=C.RESULTS_DIR)
    p.add_argument("--data", default=C.RAW_DIR)
    p.add_argument("--pairs", default=os.path.join(C.REPO_ROOT, "data", "parity"))
    p.add_argument("--target", type=float, default=0.99,
                   help="fraction of real Tasks-extracted signs that must be accepted")
    p.add_argument("--observed", default="",
                   help="comma-separated distances the APP reported for real signs "
                        "(read them off its result banner). These are the true "
                        "deployment distribution -- better evidence than any proxy.")
    p.add_argument("--set", type=float, default=None,
                   help="use this exact threshold instead of a computed one")
    p.add_argument("--apply", action="store_true", help="write the recommendation back")
    a = p.parse_args()

    observed = np.array([float(x) for x in a.observed.split(",") if x.strip()])

    from tensorflow import keras

    means, precision, current_thr, meta = load_gate(a.results)
    class_names = meta["class_names"]
    seq_len = int(meta["seq_len"])
    conf_thr = float(meta.get("confidence_threshold", 0.75))

    model = keras.models.load_model(os.path.join(a.results, "model.keras"))
    emb = ood.embedding_model(model)

    print(f"\nmodel: seq_len={seq_len} classes={len(class_names)}")
    print(f"shipped gate: reject distance > {current_thr:.3f}  (confidence >= {conf_thr})")

    def distances(clips):
        if not clips:
            return np.array([]), None
        X = batch(clips, seq_len)
        Z = emb.predict(X, verbose=0, batch_size=64)
        d, _ = ood.mahalanobis_min(Z, means, precision)
        return d, X

    # --- the three populations that matter -------------------------------
    train_clips, train_y = load_raw(a.data, class_names)
    d_train, X_train = distances(train_clips)

    tasks_clips, tasks_y = load_parity(a.pairs, class_names, "tasks")
    d_tasks, _ = distances(tasks_clips)

    hol_clips, hol_y = load_parity(a.pairs, class_names, "holistic")
    d_hol, _ = distances(hol_clips)

    d_ood = np.array([])
    if X_train is not None:
        Z_ood = emb.predict(synth_ood(X_train, train_y), verbose=0, batch_size=64)
        d_ood, _ = ood.mahalanobis_min(Z_ood, means, precision)

    print("\ndistance distributions")
    describe("training clips (Holistic)", d_train, current_thr)
    describe("parity clips (Holistic)", d_hol, current_thr)
    describe("parity clips (TASKS = the phone)", d_tasks, current_thr)
    describe("synthetic not-a-sign", d_ood, current_thr)

    # --- is the landmark source the cause? -------------------------------
    if len(d_tasks) and len(d_hol):
        shift = np.median(d_tasks) / max(np.median(d_hol), 1e-9)
        print(f"\ntasks/holistic median ratio on the SAME frames: {shift:.2f}x")
        if shift > 1.5:
            print("  -> the landmark source is the cause. Refitting the threshold on")
            print("     Tasks distances is the correct fix.")
        else:
            print("  -> the landmark source is NOT the cause; both extractions sit in the")
            print("     same place. If the phone still reports much larger distances, the")
            print("     difference is the phone pipeline (framing, rotation, fps, mirror)")
            print("     and a bigger threshold would only be hiding it.")

    if len(observed):
        describe("OBSERVED on-device (real signs)", observed, current_thr)

    # Whichever population best represents the phone drives the recommendation:
    # distances read off the app beat a webcam proxy, which beats training data.
    if len(observed):
        deploy, deploy_name = observed, "observed on-device"
    elif len(d_tasks):
        deploy, deploy_name = d_tasks, "parity Tasks clips"
    else:
        deploy, deploy_name = d_train, "training clips (POOR proxy for the phone)"
    print(f"\nrecommendation will be based on: {deploy_name} (n={len(deploy)})")

    # --- sweep -----------------------------------------------------------
    print("\nthreshold sweep")
    print(f"  {'thr':>7s}  {'accept real':>12s}  {'accept train':>13s}  {'reject OOD':>11s}  {'youden':>7s}")
    hi = float(np.ceil(max(deploy.max() if len(deploy) else 0,
                           d_ood.max() if len(d_ood) else 0))) + 2
    lo = max(0.0, float(np.floor(min(deploy.min() if len(deploy) else 0, 2.0))))
    step = max(1.0, round((hi - lo) / 24.0))
    best = None
    thr = lo
    while thr <= hi:
        ar = float((deploy <= thr).mean()) if len(deploy) else float("nan")
        at = float((d_train <= thr).mean()) if len(d_train) else float("nan")
        ro = float((d_ood > thr).mean()) if len(d_ood) else float("nan")
        # Youden's J: how much better than chance this threshold separates real
        # signs from not-signs. Its peak is the best single operating point, and
        # a peak near zero means the gate has no useful operating point at all.
        j = ar + ro - 1.0 if len(d_ood) and len(deploy) else float("nan")
        if len(d_ood) and len(deploy) and (best is None or j > best[1]):
            best = (float(thr), j, ar, ro)
        print(f"  {thr:7.1f}  {ar:11.1%}  {at:12.1%}  {ro:10.1%}  {j:7.3f}")
        thr += step

    # --- recommendation --------------------------------------------------
    if a.set is not None:
        needed = float(a.set)
        print(f"\nusing the threshold you specified: {needed:.2f}")
    else:
        needed = float(np.quantile(deploy, a.target))
        print(f"\nto accept {a.target:.0%} of real signs ({deploy_name}): "
              f"threshold >= {needed:.2f}")

    if len(d_ood):
        keep = 100.0 * float((d_ood > needed).mean())
        print(f"  at {needed:.2f}, synthetic not-a-sign rejection = {keep:.1f}%")
        # The decision that actually matters. A gate that must be opened so wide
        # to admit real signs that it no longer stops non-signs is not a safety
        # feature -- it is only a source of false rejections.
        if keep < 10.0:
            print("\n  VERDICT: at the threshold real signs require, this gate rejects")
            print(f"           almost nothing ({keep:.1f}%). It is not buying safety, only")
            print("           false rejections. Prefer the trained `none` class plus the")
            print("           confidence threshold, and set this effectively off.")
        elif keep < 40.0:
            print("\n  VERDICT: weak but non-zero discrimination. Usable as a second line")
            print("           behind the `none` class; do not rely on it alone.")
    if best:
        print(f"\nbest separation (Youden J) at threshold {best[0]:.1f}: "
              f"J={best[1]:.3f} (accept {best[2]:.0%} / reject {best[3]:.0%})")
        if best[1] < 0.2:
            print("  J below 0.2 means the two populations overlap almost completely --")
            print("  there is no threshold at which this gate works well.")

    if not len(observed) and not len(d_tasks):
        print("\nNOTE: no deployment-distribution data was available, so the sweep above")
        print("      is based on TRAINING distances, which are systematically smaller")
        print("      than the phone's. Re-run with the numbers the app shows:")
        print("        python scripts/calibrate_ood.py --observed 26.7,24.1,28.9")

    if a.apply:
        meta_path = os.path.join(a.results, "model_meta.json")
        ood_path = os.path.join(a.results, "ood.json")
        meta["ood_threshold"] = needed
        meta["ood_threshold_source"] = f"calibrated on {len(d_tasks)} Tasks clips @ p{a.target:.0%}"
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        with open(ood_path) as fh:
            gate = json.load(fh)
        gate["threshold"] = needed
        with open(ood_path, "w") as fh:
            json.dump(gate, fh)
        print(f"\napplied {needed:.3f} to model_meta.json and ood.json")
        print("Prototypes, model.keras and model.tflite are untouched. Now:")
        print("  Copy-Item results\\ood.json, results\\model_meta.json -Destination app\\assets\\models\\ -Force")
        print("  python scripts\\make_golden.py")
        print("  .\\scripts\\retrain_and_deploy.ps1 -AppOnly")


if __name__ == "__main__":
    main()
