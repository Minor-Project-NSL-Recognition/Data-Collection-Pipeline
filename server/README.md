# NSL recognition server

Serves the trained BiLSTM over HTTP + WebSocket so a Flutter client can
recognize signs without running MediaPipe on the phone.

**Why server-side.** There is no MediaPipe Holistic build for mobile, so an
on-device app would have to rebuild the 225-vector from separate Pose and Hand
detectors — a different landmark source than the one the model was trained on,
and one whose equivalence is unproven (`scripts/parity_spike.py` exists to
measure it). Running Holistic on the server sidesteps that question entirely:
the request path is the same `nslr/` code the 99.7% figure was measured on.

## Endpoints

| | |
|---|---|
| `GET /health` | backend, class count, thresholds, target fps |
| `GET /classes` | class keys + display labels |
| `POST /predict/landmarks` | `{"clip": [[225 floats] × n]}` → decision |
| `POST /predict/npy` | raw `.npy` bytes → decision (same, ~10× less wire) |
| `WS /ws/stream` | live frames → decision |

### Response

```json
{
  "status": "accepted",              // accepted | low_confidence | unknown
  "label": "need_ambulance",
  "display": "4. I need an ambulance (Medical)",
  "confidence": 0.98,
  "best_guess": "need_ambulance",    // set even when status != accepted
  "top3": [{"label": "...", "confidence": 0.98}, ...],
  "ood_distance": 4.2,               // Mahalanobis; > threshold -> unknown
  "n_frames": 95,
  "standardize": "padded"
}
```

`status` is the field to act on. `accepted` means it cleared both the softmax
threshold (0.75) and the open-set gate (distance ≤ 13.02). **Speak audio only on
`accepted`** — the 7-way softmax is closed-world and will name a phrase for
anything at all, which is what `ood_distance` exists to catch.

### WebSocket protocol

Binary messages are JPEG frames. Text messages are JSON control.

```
client → {binary JPEG}          server → {"type":"ack","accepted":true,"n_frames":42,...}
client → {"type":"done"}        server → {"type":"result", ...}
client → {"type":"reset"}       server → {"type":"reset_ok"}
client → {"type":"ping"}        server → {"type":"pong"}
```

Query params: `?mirror=0` if the client already mirrored the frame,
`?fps=` to override the decimation target.

**Stream during the sign, don't upload after it.** Landmark extraction is
~38 ms/frame and is the entire cost — a 95-frame clip is ~3.6 s of work. Sent
live, that overlaps with signing and finishes when the user does; the `done`
round-trip is then only ~15 ms. Uploading a finished video serializes the two
and adds seconds of dead air.

## Two constraints that are not tuning knobs

**Frame rate.** The model reads frame *count* as duration. The training clips
were captured at ~15.7 fps (`config.TRAIN_CAPTURE_FPS`, measured across all 570
clips), so a 30 fps client would hand it twice the frames for the same sign.
`session.py` decimates before MediaPipe — which also halves server cost and
keeps Holistic's tracker at the cadence it had during recording. Verified:
180 frames pushed at 30 fps → 92 kept at 15.3 fps.

**`model_complexity=1`.** Dropping to 0 would roughly double throughput and
produce different landmarks, silently invalidating the trained model.

## Run it

```bash
python scripts/train_model.py        # if results/model.keras doesn't exist
python scripts/export_tflite.py      # -> results/model.tflite + ood.json

pip install fastapi "uvicorn[standard]"      # additive; safe for the pinned venv
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Verify without a phone:

```bash
python scripts/test_client.py                # replays data/raw + drives the WS
```

Expect ~100% on the replay — those clips were in the training set, so this
checks that the *request path* matches the offline pipeline, not generalization.
Anything below ~95% means `server/inference.py` diverged from `nslr/`.

### Docker

```bash
cp .env.example .env                  # from the repo root
docker compose up -d                  # LAN only
docker compose --profile quick up -d  # + a random https://*.trycloudflare.com URL
docker compose --profile named up -d  # + your own stable hostname
./scripts/tunnel_url.sh               # print the quick tunnel's URL
```

The image skips TensorFlow entirely (`tflite-runtime` + the 882 KB export),
which is the difference between a ~1 GB image and a ~3 GB one. Full runbook in
[../DEMO.md](../DEMO.md).

## Authentication

Setting `NSL_API_KEY` requires that key on every endpoint except `/health`
(left open so container healthchecks and the phone's reachability test keep
working). Unset, the API is open — fine on a LAN, **not** fine behind a tunnel,
where anyone with the URL can stream frames into your MediaPipe workers.

Supply it as an `X-API-Key` header or a `?key=` query parameter. Both work; the
query form exists because WebSocket clients cannot reliably set headers on the
upgrade request, and the app uses it for that reason. Compared with
`hmac.compare_digest`.

## Deployment notes

Each WebSocket session holds its own MediaPipe graph (~150 MB), so concurrency
is memory-bound, not CPU-bound. `NSL_MAX_SESSIONS` (default 4) sheds load with a
`server_busy` error rather than an OOM kill. Scale with replicas, not with
uvicorn `--workers`.

For a live demo, run it on a laptop on the same Wi-Fi — no cold start, no
bandwidth surprises. Cloud Run scales to zero but a container this size takes
10–30 s to cold-start, which is painful in front of an audience. Bandwidth is
roughly 2–4 Mbps up at 640×480/16 fps; fine on Wi-Fi, marginal on mobile data.

Two things worth a sentence in the report: the app requires connectivity (a real
weakness for an emergency use case — on-device is the roadmap, and
`parity_spike.py` measures whether it's reachable), and camera frames leave the
device (they are processed in memory and never written to disk).
