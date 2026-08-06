"""FastAPI service for NSL phrase recognition.

Two ways in, for two different jobs:

  POST /predict/landmarks  -- already-extracted (n,225) clips. This is the
      verification path: replaying data/raw through it proves the server
      reproduces the offline pipeline exactly (scripts/test_client.py).

  WS   /ws/stream          -- the real path. The phone streams JPEG frames
      *while the user signs*, so landmark extraction (~38 ms/frame, the entire
      cost) overlaps with signing instead of following it. When the client says
      "done", only normalize + standardize + infer remain: ~15 ms. Uploading a
      finished video instead would add several seconds of dead time.

    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import asyncio
import hmac
import io
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from nslr import config as C
from server.inference import Predictor
from server.session import StreamSession

# Each live session holds its own MediaPipe graph (~150 MB). Cap concurrency
# explicitly so load sheds with a clear message instead of an OOM kill.
MAX_SESSIONS = int(os.environ.get("NSL_MAX_SESSIONS", "4"))

# Optional shared secret. Unset (the default) leaves the API open, which is fine
# on a LAN. Set it whenever the server is exposed publicly -- e.g. through a
# Cloudflare tunnel -- or anyone with the URL can stream frames into your
# MediaPipe workers. /health stays open so container healthchecks keep working.
API_KEY = os.environ.get("NSL_API_KEY", "").strip()

app = FastAPI(title="NSL Phrase Recognition", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_predictor = None
_predict_lock = threading.Lock()   # TFLite interpreters are not thread-safe
_sessions = asyncio.Semaphore(MAX_SESSIONS)


def predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def _key_ok(headers, query):
    """Accept the key from a header or a query param.

    Browsers and several WebSocket clients cannot set custom headers on the
    upgrade request, so `?key=` has to work too. Compared with compare_digest
    to avoid leaking length/prefix information through timing.
    """
    if not API_KEY:
        return True
    supplied = headers.get("x-api-key") or query.get("key") or ""
    return hmac.compare_digest(supplied, API_KEY)


def require_key(request: Request):
    if not _key_ok(request.headers, request.query_params):
        raise HTTPException(401, "missing or invalid API key")


def _predict(clip):
    with _predict_lock:
        return predictor().predict(clip)


@app.on_event("startup")
def _warmup():
    """Load the model and run one inference now, so the first real request
    doesn't pay for lazy initialisation."""
    p = predictor()
    p.predict(np.zeros((10, C.FEATURE_DIM), dtype=np.float32))
    ood_desc = "off"
    if p.ood:
        ood_desc = f"d>{p.ood[2]:.2f} ({p.ood_threshold_source})"
    print(f"ready — backend={p.backend} classes={p.class_names} seq_len={p.seq_len} "
          f"ood={ood_desc} max_sessions={MAX_SESSIONS} "
          f"auth={'on' if API_KEY else 'OFF (open)'}")


@app.get("/health")
def health():
    """Deliberately unauthenticated: container healthchecks and the phone's
    'can I even reach the server' test both need it."""
    p = predictor()
    return {"status": "ok", "backend": p.backend, "n_classes": p.n_classes,
            "seq_len": p.seq_len, "confidence_threshold": p.threshold,
            "open_set_rejection": p.ood is not None,
            "ood_threshold": p.ood[2] if p.ood else None,
            "ood_threshold_source": p.ood_threshold_source,
            "target_fps": C.TRAIN_CAPTURE_FPS, "max_sessions": MAX_SESSIONS,
            "auth_required": bool(API_KEY)}


@app.get("/classes", dependencies=[Depends(require_key)])
def classes():
    p = predictor()
    return {"classes": [{"key": k, "label": C.CLASSES.get(k, k)} for k in p.class_names]}


@app.post("/predict/landmarks", dependencies=[Depends(require_key)])
async def predict_landmarks(request: Request):
    """JSON {"clip": [[225 floats] x n_frames]} -> a decision."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(400, "body must be JSON")
    if "clip" not in body:
        raise HTTPException(400, "missing 'clip'")
    try:
        clip = np.asarray(body["clip"], dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"bad clip: {exc}")
    try:
        return await run_in_threadpool(_predict, clip)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/predict/npy", dependencies=[Depends(require_key)])
async def predict_npy(request: Request):
    """Raw .npy bytes -> a decision. Same thing, ~10x less wire overhead."""
    raw = await request.body()
    try:
        clip = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise HTTPException(400, f"could not read .npy: {exc}")
    try:
        return await run_in_threadpool(_predict, clip)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    """Binary messages are JPEG frames; text messages are JSON control.

    Control:  {"type":"done"}  {"type":"reset"}  {"type":"ping"}
    Query:    ?mirror=0 if the client already mirrored;  ?fps= to override
    """
    await ws.accept()

    if not _key_ok(ws.headers, ws.query_params):
        await ws.send_json({"type": "error", "error": "unauthorized",
                            "detail": "missing or invalid API key"})
        await ws.close(code=1008)
        return

    if _sessions.locked():
        await ws.send_json({"type": "error", "error": "server_busy",
                            "detail": f"all {MAX_SESSIONS} slots in use"})
        await ws.close()
        return

    async with _sessions:
        mirror = ws.query_params.get("mirror", "1") not in ("0", "false", "False")
        try:
            fps = float(ws.query_params.get("fps", C.TRAIN_CAPTURE_FPS))
        except ValueError:
            fps = C.TRAIN_CAPTURE_FPS

        sess = await run_in_threadpool(StreamSession, fps, mirror)
        p = predictor()
        await ws.send_json({"type": "ready", "target_fps": sess.target_fps,
                            "mirror": mirror, "classes": p.class_names})
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break

                if msg.get("bytes") is not None:
                    status = await run_in_threadpool(sess.add_jpeg, msg["bytes"])
                    await ws.send_json({"type": "ack", **status})
                    continue

                text = msg.get("text")
                if not text:
                    continue
                try:
                    cmd = json.loads(text)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "error": "bad_json"})
                    continue

                kind = cmd.get("type")
                if kind == "ping":
                    await ws.send_json({"type": "pong"})
                elif kind == "reset":
                    sess.reset()
                    await ws.send_json({"type": "reset_ok"})
                elif kind == "done":
                    clip = sess.clip()
                    stats = sess.stats()
                    if len(clip) < 5:
                        await ws.send_json({"type": "error", "error": "too_short",
                                            "detail": f"only {len(clip)} usable frames",
                                            "stats": stats})
                    else:
                        result = await run_in_threadpool(_predict, clip)
                        await ws.send_json({"type": "result", **result, "stats": stats})
                    sess.reset()
                else:
                    await ws.send_json({"type": "error", "error": "unknown_type",
                                        "detail": str(kind)})
        except WebSocketDisconnect:
            pass
        finally:
            await run_in_threadpool(sess.close)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": type(exc).__name__, "detail": str(exc)})
