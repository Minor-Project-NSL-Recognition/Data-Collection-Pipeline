# Demo runbook

Running the recognition backend in Docker and driving it from the Flutter app.

Written to be followed in order on the day. The night-before section is not
optional — every step in it is something that has no good fix once an audience
is watching.

**Two ways to reach the server.** Pick one; they are not exclusive, and the
compose file publishes port 8000 on the LAN either way so the tunnel can fail
without taking the demo with it.

| | LAN | Cloudflare tunnel |
|---|---|---|
| Setup | firewall rule | none (quick) / account + domain (named) |
| URL | `http://192.168.1.7:8000`, changes with the network | `https://…`, stable with a named tunnel |
| Phone must be on same Wi-Fi | yes | no |
| Latency | lowest | +internet round trip |
| Public | no | **yes — set an API key** |

Jump to [Cloudflare tunnel](#cloudflare-tunnel) if you want the URL-based route.

---

## The night before

### 1. Export the model (once)

The image copies `results/model.tflite`; the build fails immediately if it is
missing, which is deliberate.

```bash
python scripts/export_tflite.py
ls results/model.tflite results/model_meta.json results/ood.json
```

### 2. Configure and build

```bash
cp .env.example .env
python -c "import secrets;print(secrets.token_urlsafe(24))"   # put in NSL_API_KEY
```

Then build. From the **repo root**, not from `server/` — the Dockerfile copies
`nslr/` too.

```bash
docker compose build
```

Expect roughly 800 MB–1 GB and a few minutes. If `tflite-runtime==2.14.0` fails
to resolve for Python 3.11, swap that one line in `server/requirements.txt` for
`tensorflow-cpu==2.17.1` and rebuild — `server/inference.py` falls back to
TensorFlow's interpreter automatically. The image gets much bigger; nothing else
changes.

### 3. Start it and verify locally

```bash
docker compose up -d
docker compose logs -f server     # wait for the `ready — backend=…` line

curl http://127.0.0.1:8000/health
python scripts/test_client.py --n 40 --key <your NSL_API_KEY>
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

## Cloudflare tunnel

Gives the server a public `https://` URL, so the phone does not need to be on
your Wi-Fi at all — and no firewall rule, because the connection is outbound
from the container.

**Set `NSL_API_KEY` in `.env` before doing this.** A tunnel URL is reachable by
anyone on the internet, and without a key anyone with it can stream frames into
your MediaPipe workers. `tunnel_url.sh` warns you if you forget.

### Quick tunnel — no account, random URL

```bash
docker compose --profile quick up -d
./scripts/tunnel_url.sh
```

Prints something like `https://random-words-here.trycloudflare.com`, then checks
`/health` through it. Put that URL and your API key into the app (gear icon).

The hostname is **regenerated on every restart**, so plan for changing it on the
phone rather than rebuilding the app:

1. `bash scripts/tunnel_url.sh` — prints today's URL.
2. Get it onto the phone however you like (chat, email, notes).
3. In the app: **gear icon → paste button → Connect**.

The API key does not change with the tunnel, so it stays filled in — only the
URL needs replacing. The paste button exists because a quick tunnel hostname is
40-odd random characters and typing it on a phone is the worst part of this
workflow.

If the app was built with `build_app.ps1`, **Use built-in server** puts the
compiled-in URL back without a reinstall.

### Named tunnel — stable URL, needs a Cloudflare account

**This is the setup you want.** The URL survives restarts and reboots, and the
app can have it compiled in, so there is nothing to type on demo day.

Requires a domain on Cloudflare (a free `.dev`/cheap domain is enough). This is
the one part nobody can do for you — the tunnel lives in *your* account.

**One-time setup**

1. Cloudflare Zero Trust → **Networks → Tunnels → Create a tunnel** → *Cloudflared*.
2. Name it, then copy the **token** out of the install command it shows you.
3. Add a **public hostname**: pick `nsl.yourdomain.com`, service type **HTTP**,
   URL `server:8000` — the compose *service name*, not `localhost`. The
   cloudflared container resolves it on the compose network; `localhost` would
   point at cloudflared itself.
4. Put both values in `.env`:

   ```ini
   TUNNEL_TOKEN=eyJhIjoi...            # from step 2
   TUNNEL_HOSTNAME=nsl.yourdomain.com  # from step 3, no https://
   ```

5. Start it and confirm the whole path:

   ```powershell
   docker compose --profile named up -d
   powershell -ExecutionPolicy Bypass -File scripts\check_tunnel.ps1
   ```

   `check_tunnel.ps1` checks localhost first and then the tunnel, so a failure
   tells you which half is broken instead of just "it doesn't work".

6. Build the app with that URL baked in:

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\build_app.ps1 -Install
   ```

Now the app opens **straight to the camera** — no setup screen, no typing. The
URL and key are compiled in via `--dart-define`, read in `app/lib/main.dart`.

**After that, starting the server is one command:**

```powershell
docker compose --profile named up -d
```

`restart: unless-stopped` means it also comes back by itself after a reboot,
as long as Docker Desktop is running.

**Two things worth knowing**

- A URL saved from the gear icon **wins over the compiled-in one** — it is a
  deliberate override, and silently replacing it would be baffling. `-Install`
  uninstalls first, which clears it, so a fresh install always uses the baked
  value.
- `build_app.ps1` defaults to `--release`, not debug. The frame encoder is pure
  Dart and is the client-side bottleneck; release is AOT-compiled and markedly
  faster. Since the model reads frame count as duration, a slow client costs
  accuracy — so do not demo a debug build.

### What the tunnel changes

- **`https://` → `wss://` automatically.** The app rewrites the scheme, so no
  cleartext-HTTP config is involved. The Android network-security file becomes
  irrelevant on this path.
- **Idle WebSockets get closed.** Cloudflare drops connections that go quiet,
  and between signs this one does. The client pings every 30 s to prevent it —
  without that you would see the first sign work and a later one fail with a
  disconnect, which is a horrible thing to debug live.
- **Frames now cross the internet.** Roughly 2–4 Mbps up at 640×480/16 fps. On
  a weak uplink, drop `maxWidth` to 480 and JPEG quality to 60 in
  `app/lib/frame_encoder.dart`. Watch the `fps` chip: under ~10 and accuracy
  starts to suffer.
- **Cloudflare's free tunnel is not a production service.** It is fine for a
  demo. Do not build a submission around it staying up unattended.

---

## On the day

**Named tunnel (the whole thing):**

```powershell
docker compose --profile named up -d
powershell -ExecutionPolicy Bypass -File scripts\check_tunnel.ps1
```

Open the app and sign. Nothing to type — the URL is compiled in, and it did not
change overnight.

**Quick tunnel or LAN**, where the address moves and must be re-entered:

```powershell
docker compose up -d                  # add --profile quick
curl http://127.0.0.1:8000/health     # sanity
ipconfig                              # LAN route: confirm the IP has not changed
bash scripts/tunnel_url.sh            # quick tunnel: get today's URL
```

Confirm the URL in the app (gear icon), then tap the button, sign, and tap again.

**If the IP changed**, that alone breaks the app. It changes whenever the
network changes. Check it every single time. This is exactly the failure a named
tunnel removes.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `server` chip red | wrong IP, firewall, or different Wi-Fi | load `/health` in the phone browser — that isolates it |
| "Server rejected the API key" | key mismatch | must equal `NSL_API_KEY` in `.env`; recreate with `docker compose up -d` after changing it |
| `/health` works, app doesn't | cleartext HTTP blocked | already handled in `network_security_config.xml`; confirm the URL is `http://`, not `https://` |
| `pose` stays red | frames arriving rotated | see `frame_encoder.dart` rotation, `app/README.md` |
| `fps` under 10 | phone too slow at JPEG encoding | lower `maxWidth` in `frame_encoder.dart` |
| Everything → `unknown` | open-set gate rejecting | usually follows `pose` red — fix that first |
| `too_short` | released the button too fast | signs need ≥5 kept frames, ~1s |
| `server_busy` | more than `NSL_MAX_SESSIONS` clients | raise `NSL_MAX_SESSIONS` in `.env` (~150 MB each) |
| Container exits at start | `results/*` missing from the image | rebuild after `export_tflite.py` |
| Tunnel URL 502s | server not healthy yet | compose waits for the healthcheck; `docker compose logs server` |
| First sign works, later one disconnects | idle WebSocket dropped | the 30 s keepalive should prevent it; check `docker compose logs tunnel-*` |
| Quick tunnel URL stopped working | it changed on restart | `./scripts/tunnel_url.sh`, or use a named tunnel |

Useful during a demo:

```bash
docker compose logs -f server     # live server log
docker stats nsl-server           # memory, if several phones connect
docker compose restart server     # clears wedged sessions
docker compose ps                 # what is actually running
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
