"""Aggregate the paired clips from parity_spike.py into a go / no-go verdict.

The question this answers: if the Flutter app builds its 225-vector from
MediaPipe Tasks instead of Holistic, does the model trained on Holistic still
work? Three things decide it, and they fail in different ways:

  1. accuracy      -- does Tasks-fed prediction still match the true phrase?
  2. detection     -- do the two stacks agree on *when* a hand is present?
                      (undetected hands become zeros, which is a big input change)
  3. landmark drift -- where both detect, how far apart are the normalized values?

    python scripts/parity_report.py
    python scripts/parity_report.py --json results/parity.json
"""

import argparse
import glob
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nslr import config as C
from nslr.preprocess import normalize_hand_block, normalize_pose_block

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parity_spike import PAIRS_DIR, classify, load_predictor

BLOCKS = [("pose", C.POSE_SLICE, normalize_pose_block, 33),
          ("left_hand", C.LEFT_HAND_SLICE, normalize_hand_block, 21),
          ("right_hand", C.RIGHT_HAND_SLICE, normalize_hand_block, 21)]


def block_drift(h_clip, t_clip):
    """Per-block normalized divergence over frames where BOTH sources detected it,
    plus each source's detection rate. Frames where either is zero are excluded
    from the divergence (nothing to compare) but counted in the rates."""
    out = {}
    for name, sl, norm, n in BLOCKS:
        hb, tb = h_clip[:, sl], t_clip[:, sl]
        h_on = np.abs(hb).sum(axis=1) > 0
        t_on = np.abs(tb).sum(axis=1) > 0
        both = h_on & t_on
        rec = {"holistic_detect": float(h_on.mean()),
               "tasks_detect": float(t_on.mean()),
               "both": int(both.sum())}
        if both.any():
            d, s = [], []
            for i in np.flatnonzero(both):
                hn = norm(hb[i]).reshape(n, 3)
                tn = norm(tb[i]).reshape(n, 3)
                d.append(np.linalg.norm(hn - tn, axis=1).mean())
                s.append(np.linalg.norm(hn, axis=1).mean())
            rec["drift"] = float(np.mean(d))
            rec["drift_rel"] = float(np.mean(d) / max(np.mean(s), 1e-9))
        out[name] = rec
    return out


def main():
    p = argparse.ArgumentParser(description="Aggregate Holistic-vs-Tasks parity clips.")
    p.add_argument("--pairs", default=PAIRS_DIR)
    p.add_argument("--results", default=C.RESULTS_DIR)
    p.add_argument("--json", default=None, help="also write the full report here")
    a = p.parse_args()

    metas = sorted(glob.glob(os.path.join(a.pairs, "*", "*.json")))
    if not metas:
        raise SystemExit(f"No paired clips under {a.pairs}. Record some first:\n"
                         f"    python scripts/parity_spike.py")

    predictor, class_names, seq_len, threshold, ood_stats = load_predictor(a.results)
    rows, drifts = [], []

    for meta_path in metas:
        with open(meta_path) as fh:
            meta = json.load(fh)
        stem = meta_path[:-len(".json")]
        h_clip = np.load(stem + "__holistic.npy")
        t_clip = np.load(stem + "__tasks.npy")

        h = classify(predictor, ood_stats, h_clip, seq_len, class_names)
        t = classify(predictor, ood_stats, t_clip, seq_len, class_names)
        truth = meta["class"]
        ood_thr = ood_stats[2] if ood_stats else float("inf")

        rows.append({
            "clip": os.path.basename(stem), "truth": truth,
            "fps": meta.get("fps"), "n_frames": meta.get("n_frames"),
            "handedness_agreement": meta.get("handedness_label_agreement"),
            "holistic": {"label": h["label"], "conf": h["conf"], "dist": h["dist"],
                         "correct": h["label"] == truth,
                         "accepted": h["conf"] >= threshold and (h["dist"] or 0) <= ood_thr},
            "tasks": {"label": t["label"], "conf": t["conf"], "dist": t["dist"],
                      "correct": t["label"] == truth,
                      "accepted": t["conf"] >= threshold and (t["dist"] or 0) <= ood_thr},
            "same_prediction": h["label"] == t["label"],
            "prob_l1": float(np.abs(np.array(h["probs"]) - np.array(t["probs"])).sum()),
        })
        drifts.append(block_drift(h_clip, t_clip))

    n = len(rows)
    h_acc = np.mean([r["holistic"]["correct"] for r in rows])
    t_acc = np.mean([r["tasks"]["correct"] for r in rows])
    h_ok = np.mean([r["holistic"]["accepted"] and r["holistic"]["correct"] for r in rows])
    t_ok = np.mean([r["tasks"]["accepted"] and r["tasks"]["correct"] for r in rows])
    same = np.mean([r["same_prediction"] for r in rows])
    fps = [r["fps"] for r in rows if r["fps"]]

    print(f"\n{'='*66}\n  PARITY REPORT — {n} paired clips\n{'='*66}")
    if fps:
        print(f"  capture {np.median(fps):.1f} fps median "
              f"(training data was 15.7 median, 10.8-21.2 range)")
    print(f"\n  {'':22s} {'holistic':>10s} {'tasks':>10s}")
    print(f"  {'top-1 correct':22s} {h_acc:>9.1%} {t_acc:>10.1%}")
    print(f"  {'correct AND accepted':22s} {h_ok:>9.1%} {t_ok:>10.1%}")
    print(f"  {'same prediction':22s} {same:>21.1%}")
    print(f"  {'mean |dprob| (L1)':22s} "
          f"{np.mean([r['prob_l1'] for r in rows]):>21.3f}")

    print(f"\n  per-block landmark drift (frames where both detected)")
    print(f"  {'block':12s} {'detect: hol':>12s} {'tasks':>8s} {'drift':>9s} {'relative':>10s}")
    for name, _, _, _ in BLOCKS:
        hd = np.mean([d[name]["holistic_detect"] for d in drifts])
        td = np.mean([d[name]["tasks_detect"] for d in drifts])
        have = [d[name] for d in drifts if "drift" in d[name]]
        if have:
            dr = np.mean([x["drift"] for x in have])
            rel = np.mean([x["drift_rel"] for x in have])
            print(f"  {name:12s} {hd:>11.0%} {td:>8.0%} {dr:>9.4f} {rel:>9.1%}")
        else:
            print(f"  {name:12s} {hd:>11.0%} {td:>8.0%} {'--':>9s} {'--':>10s}")

    by_class = {}
    for r in rows:
        by_class.setdefault(r["truth"], []).append(r)
    print(f"\n  per-class (n | holistic -> tasks)")
    for cls in sorted(by_class):
        g = by_class[cls]
        print(f"  {cls:18s} {len(g):3d} | "
              f"{np.mean([x['holistic']['correct'] for x in g]):.0%} -> "
              f"{np.mean([x['tasks']['correct'] for x in g]):.0%}")

    bad = [r for r in rows if not r["tasks"]["correct"]]
    if bad:
        print(f"\n  tasks misclassifications ({len(bad)}):")
        for r in bad[:12]:
            print(f"    {r['clip']:34s} {r['truth']:17s} -> {r['tasks']['label']:17s}"
                  f" {r['tasks']['conf']:.0%}")

    # --- verdict ---
    print(f"\n{'-'*66}")
    gap = h_acc - t_acc
    if n < 20:
        print(f"  INCONCLUSIVE — only {n} clips. Record >=5 per phrase (~35 total).")
    elif t_ok >= 0.90 and gap <= 0.05:
        print("  GO — Tasks landmarks transfer. Build the on-device Flutter path.")
    elif t_ok >= 0.75:
        print("  MARGINAL — usable but degraded. Options: raise the confidence")
        print("  threshold, or fine-tune the model on Tasks-extracted clips.")
    else:
        print(f"  NO-GO — Tasks accuracy {t_ok:.0%} vs Holistic {h_ok:.0%}.")
        print("  Either re-record the dataset through the Tasks extractor, or")
        print("  fall back to the Flutter-client + Python-server architecture.")
    print(f"{'-'*66}\n")

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump({"n_clips": n, "holistic_accuracy": float(h_acc),
                       "tasks_accuracy": float(t_acc), "agreement": float(same),
                       "clips": rows}, fh, indent=2)
        print(f"  full report -> {a.json}")


if __name__ == "__main__":
    main()
