# Retrain → ship to the phone

Every command from new recordings (or new hyperparameters) to a working offline APK
on a connected device. PowerShell syntax; run from the repo root with the venv
active.

```powershell
.venv\Scripts\activate
```

> **`&&` does not work in Windows PowerShell 5.1.** Run these one line at a time,
> and check each one succeeded before continuing — a failure halfway through leaves
> `results/` internally inconsistent, which is the single most expensive mistake
> available here (see [Why order matters](#why-the-order-matters)).

---

## 0. Back up what git will not save you from

`results/model.keras` and `results/model.tflite` are **gitignored**. If a retrain
produces a worse model, git cannot get the old one back.

```powershell
Copy-Item results\model.keras, results\model.tflite, results\model_meta.json, results\ood.json, results\ood_stats.npz -Destination ..\nsl-backup\ -Force
```

## 1. Derive the sequence length

Only when the **dataset** changed. Skip for a hyperparameter-only change.

```powershell
python scripts\find_seq_len.py
```

Writes `data/processed/seq_len.json` at the 95th percentile of clip lengths. This
value has already moved 151 → 137 → 146 across the project, and **everything
downstream follows it**. Nothing in the Python or Dart code hardcodes it.

To pin it instead (e.g. to keep an existing model usable), edit
`data/processed/seq_len.json` by hand — `build_dataset.py` has no `--seq-len` flag
and reads that file.

## 2. Build the tensors

```powershell
python scripts\build_dataset.py
```

Writes `X.npy`, `mask.npy`, `y.npy`, `label_map.json`, `manifest.csv` to
`data/processed/`. Confirm the printed `SEQ_LEN` and per-class counts look right
before continuing.

## 3. Measure reportable accuracy

```powershell
python -u scripts\train_eval.py --verbose 2
```

Leave-one-signer-out across every eligible signer, so it trains the model N times —
**~10–15 minutes for 8 signers.** Writes `results/metrics.json` and the curve/confusion
PNGs, and saves **no model**.

This is the number for your report. Do not quote `train_model.py`'s "best val
accuracy" as a signer-independent result — that split is drawn from signers the model
trained on.

## 4. Train the deployable model

```powershell
python -u scripts\train_model.py --verbose 2
```

Writes, all in one run and all mutually dependent:

- `results/model.keras`
- `results/model_meta.json` — `class_names`, `seq_len`, `confidence_threshold`, `ood_threshold`
- `results/ood_stats.npz` — Mahalanobis prototypes + shared precision

Sanity-check the printed line **`real signs wrongly rejected: ~1%`**. If it is much
higher, the open-set gate is misfitted and the app will answer `unknown` to
everything.

## 5. Export for the phone

```powershell
python scripts\export_tflite.py
```

Writes `results/model.tflite` and `results/ood.json`, and stamps
`tflite_verified` into `model_meta.json`. It refuses to write a file that disagrees
with Keras, so **`argmax agreement 100%` must appear.**

## 6. Copy the three assets into the app

```powershell
Copy-Item results\model.tflite, results\ood.json, results\model_meta.json -Destination app\assets\models\ -Force
```

The two MediaPipe `.task` bundles in `app/assets/models/` never change — copy them
again only if you re-download them from `models/tasks/`.

## 7. Re-pin the Dart port to the new model

**Do not skip this.** The app re-implements `nslr/preprocess.py` and `nslr/ood.py` in
Dart. A mismatch does not throw — the model just receives features it never trained
on and answers confidently. This is the only thing that catches it.

```powershell
python scripts\make_golden.py
cd app
flutter test
```

All tests must pass. `flutter test` also asserts that `app/assets/models/ood.json`
came from the same run as the fixture, so it catches a half-finished step 6.

## 8. Build

```powershell
flutter analyze
flutter build apk --release
```

## 9. Install on the connected phone

```powershell
flutter devices
flutter install --release
```

⚠️ **`flutter install` only ever installs `build\app\outputs\flutter-apk\app-release.apk`.**
If you built with `--split-per-abi`, that file is **not refreshed** and you will
silently install an older build. This has already cost debugging time once. Either
build without `--split-per-abi` (as above), or install the per-ABI file directly:

```powershell
$adb = "$env:LOCALAPPDATA\Android\sdk\platform-tools\adb.exe"
flutter build apk --release --split-per-abi
& $adb install -r -d build\app\outputs\flutter-apk\app-arm64-v8a-release.apk
```

Split builds carry an ABI-offset version code (2001), so going back to a fat APK
afterwards needs `-d` to allow the downgrade, or an uninstall.

## 10. Check it on the device

Open the app — it starts in **offline** mode. Record one sign.

| Telemetry | Healthy |
|---|---|
| `offline`, `gate` | both green |
| `pose`, `hands` | green while signing |
| `cam` → `conv` → `frames` | all three climbing together |
| `fps` | ≥ 14 |
| `detect` | comfortably under 62 ms |
| `busy-drop` | low or absent |

Whichever of `cam` / `conv` / `frames` stops climbing is the failing stage. An `err`
chip or a red **Frame pipeline failed** banner shows the verbatim error.

If landmarking cannot keep up, lower `maxWidth` from 480 in
[app/lib/rgba_converter.dart](app/lib/rgba_converter.dart).

---

## Why the order matters

`model.keras`, `model.tflite`, `ood_stats.npz`, `ood.json` and `model_meta.json` are
**one atomic set**. The OOD prototypes live in the model's penultimate embedding
space, so prototypes from a different training run describe a space this model does
not have — and the result is not a crash but wholesale rejection of real signs
(measured once at 40–100% rejected, versus the intended ~1%).

Steps 4 → 5 → 6 → 7 must therefore run as a block after any retrain.

**The trap that caused this once:** `ood.json`, `ood_stats.npz`, `model_meta.json`
and `metrics.json` are **tracked in git**, while `model.keras` and `model.tflite` are
**gitignored**. So `git pull` can hand you a teammate's gate fitted on a model you do
not have. After any pull that touches `results/`, re-run steps 4–7 locally rather
than trusting the files.

## If the class list changed

Adding or removing a phrase needs three edits beyond this pipeline:

1. `nslr/config.py` → `CLASSES`
2. `app/lib/local_recognizer.dart` → `kClassLabels` (a missing key falls back to the
   raw folder name, which is what gets spoken aloud)
3. Re-record or re-check `data/parity/` if the landmark source is also in question

`class_names` itself flows automatically from `model_meta.json`; only the
human-readable labels are hardcoded.

## Quick reference

| Change | Steps |
|---|---|
| New/more recordings | 0 → 10 (all) |
| Hyperparameters only | 0, 3–10 (skip 1–2) |
| Re-export an existing model | 5 → 10 |
| After `git pull` touching `results/` | 4 → 10 |
| Dart/Kotlin change only | 8 → 10 |
