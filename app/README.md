# NSL Recognition — Flutter client

Streams camera frames to the recognition server (`../server/`) while the user
signs, then speaks the recognized phrase.

Nothing is recognized on the device: the phone is a camera and a speaker. All
MediaPipe and model work happens server-side, which is what keeps the request
path identical to the validated offline pipeline. See `../server/README.md`.

## Layout

| File | |
|---|---|
| `lib/main.dart` | app shell + server-URL setup screen |
| `lib/nsl_client.dart` | the `WS /ws/stream` protocol |
| `lib/frame_encoder.dart` | camera frame → upright JPEG, on a background isolate |
| `lib/sign_page.dart` | camera preview, hold-to-sign, telemetry, TTS |

## Run it

```bash
# 1. Start the server on your laptop, bound to all interfaces
cd ..
uvicorn server.app:app --host 0.0.0.0 --port 8000

# 2. Find the laptop's LAN address
ipconfig                      # Windows -> IPv4 Address, e.g. 192.168.1.7
hostname -I                   # Linux/Mac

# 3. Run the app on a phone on the SAME Wi-Fi
cd app
flutter devices
flutter run -d <device-id>
```

On first launch the app asks for the server URL. Enter `http://<laptop-ip>:8000`
— **not** `127.0.0.1`, which on the phone means the phone. Change it later via
the gear icon.

Then hold the button, perform one sign, release. The result appears and is
spoken if it was accepted.

## Reading the telemetry row

Those chips are the debugging surface. They come from the server's per-frame
acks, so they report what the *server* sees, not what the phone thinks it sent.

| Chip | Green means |
|---|---|
| `server` | WebSocket connected (tap when red to retry) |
| `pose` | the server is detecting a body in your frames |
| `hands` | the server is detecting at least one hand |
| `frames` | frames the server kept after decimating to ~15.7 fps |
| `fps` | frames per second the phone is actually managing to send |

**`pose` red while you are clearly in frame almost always means rotation.**
Android delivers camera frames in sensor orientation, so a portrait-locked app
must rotate them or MediaPipe sees a person lying sideways and detects nothing.
`frame_encoder.dart` rotates by `sensorOrientation`; if a particular device
disagrees, that is the value to adjust.

**`fps` below ~10** means the phone can't keep up with JPEG encoding, and
accuracy will suffer — the model was trained on clips captured at 10.8–21.2 fps
and reads frame count as duration. Lower `maxWidth` in `frame_encoder.dart`
(480 is still reasonable; below that hand landmarks degrade) or drop
`ResolutionPreset.medium` to `.low`.

## Status, and when to speak

The server returns `accepted` / `low_confidence` / `unknown`, and the app speaks
**only on `accepted`**. This is not caution for its own sake: the 7-way softmax
is closed-world and will confidently name a phrase for any input at all,
including random motion. `unknown` is the open-set gate (Mahalanobis distance in
the model's embedding space) rejecting inputs far from every known sign. An
emergency app that announces "I need an ambulance" because someone waved is
worse than one that says nothing.

## Known limitations

- **Requires a network.** Discussed in `../server/README.md`; on-device is the
  roadmap, not this build.
- **English TTS.** `flutter_tts` uses the device voice, and `ne-NP` is missing
  on most Android devices. For Nepali, bundle pre-recorded audio in
  `assets/audio/` and swap `flutter_tts` for `just_audio` — the phrase set is
  fixed at seven, so seven files cover it.
- **Debug APK is ~147 MB.** That is debug tooling, not the app.
  `flutter build apk --release --split-per-abi` produces something normal.
- **Kotlin incremental compilation is disabled** in `android/gradle.properties`
  to work around a cache-locking failure on this machine. Harmless; see the
  comment there.
- **Untested on a physical device.** It compiles and analyzes clean, and the
  server side is verified end to end, but nobody has yet held a phone in front
  of a camera and signed. Expect the rotation constant to be the first thing
  that needs attention.
