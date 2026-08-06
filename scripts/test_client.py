"""Exercise the running server without needing a phone.

Two checks, answering different questions:

  clips  -- replay real clips from data/raw through POST /predict/npy and score
            them. This verifies the SERVER PLUMBING (normalize -> standardize ->
            TFLite -> OOD) reproduces the offline pipeline. It is not a
            generalization number: these clips were in the training set, so the
            expected result is ~100% and anything lower means the request path
            diverged from nslr/.

  stream -- drive the WebSocket protocol with synthetic frames: acks, the fps
            decimation accumulator, reset, and the too-short guard. Needs the
            `websockets` package; skipped with a note if it isn't installed.

    python scripts/test_client.py                       # both, 40 clips
    python scripts/test_client.py --n 200 --only clips
    python scripts/test_client.py --url http://192.168.1.7:8000
"""

import argparse
import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from nslr import config as C


def check_health(url):
    r = requests.get(f"{url}/health", timeout=30)
    r.raise_for_status()
    h = r.json()
    print(f"  backend={h['backend']}  classes={h['n_classes']}  seq_len={h['seq_len']}")
    print(f"  threshold={h['confidence_threshold']}  open_set={h['open_set_rejection']}  "
          f"target_fps={h['target_fps']}")
    if h.get("auth_required"):
        print("  auth: required — pass --key")
    return h


def test_clips(url, raw_dir, n, seed, key=None):
    paths = sorted(glob.glob(os.path.join(raw_dir, "*", "*.npy")))
    if not paths:
        print(f"  ! no clips under {raw_dir} — skipping")
        return None
    random.Random(seed).shuffle(paths)
    paths = paths[:n]

    correct, accepted, rejected, wrong = 0, 0, 0, []
    per_class = {}
    latencies = []

    for path in paths:
        truth = os.path.basename(os.path.dirname(path))
        with open(path, "rb") as fh:
            blob = fh.read()
        headers = {"Content-Type": "application/octet-stream"}
        if key:
            headers["X-API-Key"] = key
        t0 = time.perf_counter()
        r = requests.post(f"{url}/predict/npy", data=blob, headers=headers, timeout=60)
        latencies.append((time.perf_counter() - t0) * 1000)
        if r.status_code == 401:
            raise SystemExit("401 unauthorized — the server needs --key <NSL_API_KEY>")
        if r.status_code != 200:
            print(f"  ! {os.path.basename(path)} -> HTTP {r.status_code}: {r.text[:120]}")
            continue
        res = r.json()

        hit = res["best_guess"] == truth
        correct += hit
        accepted += res["status"] == "accepted"
        rejected += res["status"] == "unknown"
        bucket = per_class.setdefault(truth, [0, 0])
        bucket[0] += hit
        bucket[1] += 1
        if not hit:
            wrong.append((os.path.basename(path), truth, res["best_guess"],
                          res["confidence"], res["ood_distance"]))

    total = len(paths)
    print(f"\n  {total} clips replayed")
    print(f"    top-1 correct        {correct}/{total} = {correct/total:.1%}")
    print(f"    accepted (conf+ood)  {accepted}/{total} = {accepted/total:.1%}")
    print(f"    rejected as unknown  {rejected}/{total}")
    print(f"    latency  median {sorted(latencies)[len(latencies)//2]:.0f} ms  "
          f"max {max(latencies):.0f} ms")

    print(f"\n    per class:")
    for cls in sorted(per_class):
        hit, tot = per_class[cls]
        print(f"      {cls:18s} {hit:3d}/{tot:<3d} {hit/tot:6.1%}")

    if wrong:
        print(f"\n    misclassified ({len(wrong)}):")
        for name, truth, got, conf, dist in wrong[:10]:
            d = f"{dist:.1f}" if dist is not None else "-"
            print(f"      {name:38s} {truth:17s} -> {got:17s} {conf:.0%}  d={d}")

    acc = correct / total
    if acc < 0.95:
        print(f"\n  !! {acc:.1%} on training clips means the server path diverged from")
        print(f"     the offline pipeline. Compare server/inference.py against")
        print(f"     scripts/live_demo.py — normalization or seq_len is the usual cause.")
    else:
        print(f"\n  server reproduces the offline pipeline ({acc:.1%} on seen clips, as expected)")
    return acc


def test_stream(url, target_fps, key=None):
    try:
        import websockets  # noqa: F401
    except ImportError:
        print("  ! `websockets` not installed — skipping the WebSocket test.")
        print("    pip install websockets     (pure Python, safe for the pinned venv)")
        return None

    import asyncio

    import cv2
    import numpy as np
    import websockets as ws_mod

    ws_url = url.replace("http://", "ws://").replace("https://", "wss://") + "/ws/stream"
    if key:
        ws_url += f"?key={key}"
    # A synthetic frame detects no landmarks, which is fine: this test is about
    # the protocol and the rate limiter, not about recognition quality.
    frame = np.full((480, 640, 3), 40, np.uint8)
    cv2.circle(frame, (320, 240), 90, (200, 180, 160), -1)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    assert ok
    blob = buf.tobytes()
    print(f"  synthetic frame: {len(blob)/1024:.1f} KB "
          f"(~{len(blob)*target_fps/1024:.0f} KB/s at {target_fps:g} fps)")

    async def run():
        async with ws_mod.connect(ws_url, max_size=4 << 20) as ws:
            ready = json.loads(await ws.recv())
            print(f"  ready: target_fps={ready['target_fps']} mirror={ready['mirror']}")

            # Push 6 s worth at 30 fps -- twice the target -- and confirm the
            # server decimates to roughly target_fps rather than keeping all.
            #
            # Send and receive must run concurrently. Awaiting each ack before
            # sending the next frame paces the client at the server's round-trip
            # instead of 30 fps, so the rate limiter never gets stressed and the
            # test passes trivially.
            N, SEND_FPS = 180, 30
            state = {"kept": 0, "acks": 0}

            async def producer():
                start = time.perf_counter()
                for i in range(N):
                    await ws.send(blob)
                    target = start + (i + 1) / SEND_FPS
                    delay = target - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(delay)

            async def consumer():
                while state["acks"] < N:
                    msg = json.loads(await ws.recv())
                    if msg.get("type") != "ack":
                        continue
                    state["acks"] += 1
                    state["kept"] = msg.get("n_frames", state["kept"])

            t0 = time.perf_counter()
            await asyncio.gather(producer(), asyncio.wait_for(consumer(), timeout=90))
            elapsed = time.perf_counter() - t0
            kept = state["kept"]
            print(f"  sent {N} frames in {elapsed:.1f}s ({N/elapsed:.1f} fps), "
                  f"server kept {kept} ({kept/elapsed:.1f} fps)")
            if N / elapsed < SEND_FPS * 0.8:
                print(f"  !  client only reached {N/elapsed:.1f} fps — decimation not "
                      f"actually stressed; treat this result as inconclusive")
            if kept > 0 and abs(kept / elapsed - target_fps) > 4:
                print(f"  !! decimation is off — expected ~{target_fps:g} fps kept")
            else:
                print(f"  decimation OK (target {target_fps:g} fps)")

            await ws.send(json.dumps({"type": "done"}))
            res = json.loads(await ws.recv())
            if res["type"] == "result":
                print(f"  result: status={res['status']} best={res['best_guess']} "
                      f"conf={res['confidence']:.0%} d={res['ood_distance']:.1f}")
                print(f"          stats={res['stats']}")
                if res["status"] != "unknown":
                    print("  note: a blank frame ideally lands as 'unknown' — it has no landmarks")
            else:
                print(f"  {res}")

            await ws.send(json.dumps({"type": "reset"}))
            print(f"  reset -> {json.loads(await ws.recv())}")

            await ws.send(blob)
            await ws.recv()
            await ws.send(json.dumps({"type": "done"}))
            short = json.loads(await ws.recv())
            print(f"  too-short guard -> {short.get('error')} ({short.get('detail')})")

    asyncio.run(run())
    return True


def main():
    p = argparse.ArgumentParser(description="Smoke-test the NSL server.")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--raw", default=C.RAW_DIR)
    p.add_argument("--n", type=int, default=40, help="clips to replay")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", choices=["clips", "stream"], default=None)
    p.add_argument("--key", default=os.environ.get("NSL_API_KEY") or None,
                   help="API key if the server sets NSL_API_KEY (defaults to $NSL_API_KEY)")
    a = p.parse_args()

    url = a.url.rstrip("/")
    print(f"=== health  {url} ===")
    try:
        health = check_health(url)
    except requests.RequestException as exc:
        raise SystemExit(f"Cannot reach {url}: {exc}\n"
                         f"Start it with:  uvicorn server.app:app --port 8000")

    if a.only != "stream":
        print(f"\n=== replay clips ===")
        test_clips(url, a.raw, a.n, a.seed, a.key)
    if a.only != "clips":
        print(f"\n=== websocket protocol ===")
        test_stream(url, health["target_fps"], a.key)


if __name__ == "__main__":
    main()
