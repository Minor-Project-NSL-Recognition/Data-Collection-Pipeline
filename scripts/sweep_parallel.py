"""Run the architecture sweep candidates concurrently instead of one after another.

scripts/sweep_architectures.py trains one candidate at a time, and each candidate
is a small model on a small batch -- on a 24-core machine that leaves most of the
CPU idle. The candidates are completely independent (separate models, separate
LOSO loops, no shared state), so nothing about the experiment changes if they run
side by side.

Measured on this repo's data (840 clips, seq_len 146, batch 16):

    sequential          1.77 s/epoch, one at a time   ~90 min for 7 candidates
    7-way parallel      4.05 s/epoch each             ~35 min for 7 candidates

Each candidate runs as its own `python scripts/sweep_architectures.py --candidates
<name>` subprocess writing into its own results directory, so the per-candidate
CSVs can't clobber each other. When they finish, this script merges them into the
usual results/architecture_sweep.csv + .md using sweep_architectures.write_results,
so the final table is byte-identical in format to a sequential run.

    python -u scripts/sweep_parallel.py                        # all candidates
    python -u scripts/sweep_parallel.py --threads 3            # if the machine feels choked
    python -u scripts/sweep_parallel.py --max-parallel 4       # if RAM is tight (~1.2 GB/proc)
    python -u scripts/sweep_parallel.py --candidates current bigger smaller
"""

import argparse
import csv
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nslr import config as C
from scripts.sweep_architectures import CANDIDATES, write_results

SWEEP_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_architectures.py")

# Every process emits the same handful of startup/retracing lines, and at 7x that
# drowns out the actual progress. Dropped: TF C++ info lines, the oneDNN AVX-512
# notice, and Keras's retracing warning (expected here -- predict() is called once
# per LOSO fold and the folds have different row counts). Anything else, including
# real warnings and tracebacks, still comes through.
_NOISE = re.compile(
    r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+: I "
    r"|oneDNN custom operations"
    r"|oneDNN supports DT_BOOL only"
    r"|triggered tf\.function retracing"
    r"|^For \(1\), please define|^more details\.$")

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


# CSV round-trips everything as strings; write_results formats these as numbers
# (and treats lstm2/dense of 0 as "no such layer", which the string "0" would not be).
_COERCE = {"lstm1": int, "lstm2": int, "dense": int, "n_params": int,
           "dropout": float, "mean_accuracy": float, "std_accuracy": float,
           "mean_epochs": float}


def read_candidate_row(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    row = rows[0]
    return {k: _COERCE.get(k, str)(v) for k, v in row.items()}


def run_candidate(name, args, work_dir, state):
    """Train one candidate in a subprocess, streaming its output with a prefix."""
    out_dir = os.path.join(work_dir, name)
    # Wipe first: a stale CSV from an earlier run must not be merged as if it
    # were fresh, should this subprocess die before writing its own.
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [sys.executable, "-u", SWEEP_SCRIPT,
           "--candidates", name,
           "--results", out_dir,
           "--processed", args.processed,
           "--epochs", str(args.epochs),
           "--batch-size", str(args.batch_size),
           "--threads", str(args.threads)]
    if not args.deterministic:
        cmd.append("--no-deterministic")

    t0 = time.perf_counter()
    log(f"[{name}] started")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=C.REPO_ROOT)
    with state["lock"]:
        state["procs"][name] = proc

    for line in proc.stdout:
        line = line.rstrip()
        if line and not _NOISE.search(line):
            log(f"[{name}] {line}")
    rc = proc.wait()
    dt = (time.perf_counter() - t0) / 60

    row = None
    if rc == 0:
        row = read_candidate_row(os.path.join(out_dir, "architecture_sweep.csv"))
    with state["lock"]:
        state["procs"].pop(name, None)
        if row is None:
            state["failed"].append(name)
            log(f"[{name}] FAILED (exit {rc}) after {dt:.1f} min")
        else:
            state["rows"].append(row)
            state["done"].append(name)
            log(f"[{name}] done in {dt:.1f} min -> "
                f"{row['mean_accuracy']:.3f} +/- {row['std_accuracy']:.3f}")


def heartbeat(state, total, stop_event, every=120):
    """Candidates print only a handful of lines each, so without this a 35-minute
    run looks hung."""
    t0 = time.perf_counter()
    while not stop_event.wait(every):
        with state["lock"]:
            running = sorted(state["procs"])
            n_done, n_failed = len(state["done"]), len(state["failed"])
        log(f"    ... {(time.perf_counter()-t0)/60:5.1f} min elapsed | "
            f"{n_done}/{total} done"
            + (f", {n_failed} failed" if n_failed else "")
            + (f" | running: {', '.join(running)}" if running else ""))


def backup_existing(results_dir):
    for base in ("architecture_sweep.csv", "architecture_sweep.md"):
        path = os.path.join(results_dir, base)
        if os.path.exists(path):
            shutil.copy2(path, path + ".bak")
            log(f"Existing {base} backed up to {base}.bak")


def main():
    p = argparse.ArgumentParser(
        description="Run architecture-sweep candidates concurrently, then merge the tables.")
    p.add_argument("--processed", default=C.PROCESSED_DIR)
    p.add_argument("--results", default=C.RESULTS_DIR)
    p.add_argument("--work-dir", default=None,
                   help="where per-candidate results land (default: <results>/_sweep_parallel)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
    p.add_argument("--threads", type=int, default=4,
                   help="intra-op threads per candidate process (default 4, measured best "
                        "on a 24-core i9 running 7 candidates at once)")
    p.add_argument("--max-parallel", type=int, default=None,
                   help="how many candidates train at once (default: as many as fit, "
                        "given --threads and the core count)")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()

    unknown = [n for n in a.candidates if n not in CANDIDATES]
    if unknown:
        p.error(f"unknown candidate(s): {unknown}. Known: {list(CANDIDATES)}")

    work_dir = a.work_dir or os.path.join(a.results, "_sweep_parallel")
    os.makedirs(a.results, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    max_parallel = a.max_parallel or min(len(a.candidates),
                                         max(1, (os.cpu_count() or 4) // a.threads))
    log(f"{len(a.candidates)} candidates, {max_parallel} at a time, "
        f"{a.threads} threads each: {a.candidates}")
    log(f"Per-candidate output -> {work_dir}\n")
    backup_existing(a.results)

    state = {"lock": threading.Lock(), "rows": [], "done": [], "failed": [], "procs": {}}
    work = queue.Queue()
    for name in a.candidates:
        work.put(name)

    def worker():
        while True:
            try:
                name = work.get_nowait()
            except queue.Empty:
                return
            try:
                run_candidate(name, a, work_dir, state)
            except Exception as exc:                      # keep one bad candidate
                with state["lock"]:                       # from killing the sweep
                    state["failed"].append(name)
                log(f"[{name}] runner error: {exc}")

    stop_event = threading.Event()
    hb = threading.Thread(target=heartbeat, args=(state, len(a.candidates), stop_event),
                          daemon=True)
    hb.start()

    t_start = time.perf_counter()
    workers = [threading.Thread(target=worker, daemon=True) for _ in range(max_parallel)]
    try:
        for t in workers:
            t.start()
        for t in workers:
            t.join()
    except KeyboardInterrupt:
        log("\nInterrupted -- stopping candidates still running; "
            "finished ones are still merged below.")
        with state["lock"]:
            for proc in list(state["procs"].values()):
                proc.terminate()
        for t in workers:
            t.join(timeout=30)
    finally:
        stop_event.set()

    total_min = (time.perf_counter() - t_start) / 60

    if not state["rows"]:
        log(f"\nNo candidate produced results ({total_min:.1f} min). "
            f"Check the output above; {a.results}/architecture_sweep.csv was left alone.")
        return 1

    csv_path, md_path = write_results(state["rows"], a.results)
    log("\n=== Ranked by mean LOSO accuracy ===")
    for r in sorted(state["rows"], key=lambda r: r["mean_accuracy"], reverse=True):
        marker = " <-- current" if r["name"] == "current" else ""
        log(f"  {r['name']:<14} {r['mean_accuracy']:.3f} +/- {r['std_accuracy']:.3f}  "
            f"({r['n_params']:,} params){marker}")
    if state["failed"]:
        log(f"\nMissing from the table (failed or interrupted): {sorted(state['failed'])}")
    log(f"\nSaved {csv_path}\nSaved {md_path}")
    log(f"Total wall time: {total_min:.1f} min for {len(state['rows'])}/{len(a.candidates)} "
        f"candidates")
    return 0 if not state["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
