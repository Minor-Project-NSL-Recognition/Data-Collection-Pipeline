# Demo runbook

Running the recognition backend in Docker and driving it from the Flutter app.

Written to be followed in order on the day. The night-before section is not
optional — every step in it is something that has no good fix once an audience
is watching.

---

## The night before

### 1. Export the model (once)

The image copies `results/model.tflite`; the build fails immediately if it is
missing, which is deliberate.

```bash
python scripts/export_tflite.py
ls results/model.tflite results/model_meta.json results/ood.json
```

### 2. Build the image

From the **repo root**, not from `server/` — the Dockerfile copies `nslr/` too.

```bash
docker build -f server/Dockerfile -t nsl-server .
```

Expect roughly 800 MB–1 GB and a few minutes. If `tflite-runtime==2.14.0` fails
to resolve for Python 3.11, swap that one line in `server/requirements.txt` for
`tensorflow-cpu==2.17.1` and rebuild — `server/inference.py` falls back to
TensorFlow's interpreter automatically. The image gets much bigger; nothing else
changes.

### 3. Start it and verify locally

```bash
docker run -d --name nsl -p 8000:8000 --restart unless-stopped nsl-server
docker logs -f nsl          # wait for the uvicorn startup line, Ctrl-C to detach

curl http://127.0.0.1:8000/health
python scripts/test_client.py --n 40
```

`test_client.py` should report ~100% on the replayed clips. Anything lower means
the image is not serving the pipeline you think it is — fix it now, not
tomorrow.

### 4. Open the firewall

**This is the step that breaks demos.** The container publishes on all
interfaces, but Windows blocks the inbound connection, so the laptop works and
the phone silently cannot connect.

PowerShell, as Administrator:

```powershell
New-NetFirewallRule -DisplayName "NSL demo 8000" -Direction Inbound `
  -Protocol TCP -LocalPort 8000 -Action Allow
```

To remove it afterwards: `Remove-NetFirewallRule -DisplayName "NSL demo 8000"`.

### 5. Prove it from the phone

```bash
ipconfig                    # IPv4 Address, e.g. 192.168.1.7
```

Open `http://192.168.1.7:8000/health` **in the phone's browser**. You want the
JSON. If it does not load, the app will not work either, and you have found out
while there is still time to fix it.

Then set that URL in the app and do one real sign end to end.

### 6. Prepare for no internet

Venue Wi-Fi may not exist or may isolate clients from each other. Two hedges:

```bash
# Carry the image as a file in case the demo machine can't pull/build
docker save nsl-server | gzip > nsl-server.tar.gz
# On the demo machine:  gunzip -c nsl-server.tar.gz | docker load
```

And bring a phone hotspot. Connect the laptop to the *phone's* hotspot — then
both are on the same network with no venue infrastructure involved. Re-check
`ipconfig`, the IP will have changed.

---

## On the day

```bash
docker start nsl                      # or `docker run` per step 3
curl http://127.0.0.1:8000/health     # sanity
ipconfig                              # confirm the IP has not changed
```

Confirm the URL in the app (gear icon), then hold the button and sign.

**If the IP changed**, that alone breaks the app. It changes whenever the
network changes. Check it every single time.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `server` chip red | wrong IP, firewall, or different Wi-Fi | load `/health` in the phone browser — that isolates it |
| `/health` works, app doesn't | cleartext HTTP blocked | already handled in `network_security_config.xml`; confirm the URL is `http://`, not `https://` |
| `pose` stays red | frames arriving rotated | see `frame_encoder.dart` rotation, `app/README.md` |
| `fps` under 10 | phone too slow at JPEG encoding | lower `maxWidth` in `frame_encoder.dart` |
| Everything → `unknown` | open-set gate rejecting | usually follows `pose` red — fix that first |
| `too_short` | released the button too fast | signs need ≥5 kept frames, ~1s |
| `server_busy` | more than `NSL_MAX_SESSIONS` clients | `docker run -e NSL_MAX_SESSIONS=8 ...` (~150 MB each) |
| Container exits at start | `results/*` missing from the image | rebuild after `export_tflite.py` |

Useful during a demo:

```bash
docker logs -f nsl        # live server log
docker stats nsl          # memory, if several phones connect
docker restart nsl        # clears wedged sessions
```

---

## Fallback if Docker misbehaves

The server runs perfectly well without it, from the pinned `.venv`:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` is what makes it reachable from the phone; the default
`127.0.0.1` is not. This path uses TensorFlow's TFLite interpreter instead of
`tflite-runtime` — same model, same numbers.

---

## Two questions worth having answers ready for

**"Why does an emergency app need the internet?"** It doesn't, in principle —
this build puts inference on a server so the request path stays identical to the
pipeline the accuracy was measured on. On-device is the roadmap;
`scripts/parity_spike.py` exists to measure whether the mobile landmark stack is
equivalent, since MediaPipe Holistic has no mobile build.

**"What happens if it sees a sign it doesn't know?"** It says so. There is a
7th `none` class plus an open-set gate that measures Mahalanobis distance to
each class prototype in the model's embedding space and rejects anything far
from all of them. Worth demonstrating deliberately: wave at it and let it
answer "Unknown sign". A closed-world softmax would have confidently named an
emergency.
