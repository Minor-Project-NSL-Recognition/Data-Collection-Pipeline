# NSL Emergency-Phrase Recognition — Data & Model Pipeline

Landmark-based recognition of six Nepali Sign Language (NSL) emergency phrases
(MediaPipe Holistic → dual-anchor normalization → BiLSTM). This repo is the
offline data + training side of a larger project; `main.pdf` (the full
proposal) is kept outside this repo, so code comments referencing "proposal
X.Y" won't resolve to a local file.

The current committed model reaches **99.7% signer-independent accuracy**
across 6 classes — see [Current results](#current-results).

## The six phrases

| Key | GUI label |
|---|---|
| `cant_breathe` | 1. I can't breathe (Medical) |
| `building_on_fire` | 2. The building is on fire (Fire) |
| `call_police` | 3. Call the police (Crime) |
| `need_ambulance` | 4. I need an ambulance (Medical) |
| `help_danger` | 5. Help me / I am in danger (Generic) |
| `need_toilet` | 6. I need to go to the toilet (Basic need) |

(Phrase #6 is `need_toilet` by design — the proposal text still says
"earthquake"; the data/code here is the source of truth.)

## Layout

```
nslr/                  importable package — single source of truth
  config.py            locked constants, class list, paths, seq_len fallback
  landmarks.py         MediaPipe result -> 225-vector
  preprocess.py        dual-anchor normalization + length standardization
  dataset.py           raw clips -> X/mask/y ; load back for training
  model.py             the BiLSTM architecture (build_bilstm)
scripts/               thin entrypoints (import nslr)
  record.py            GUI landmark recorder (webcam -> raw clips)
  find_seq_len.py      derive SEQ_LEN from the recorded frame-count distribution
  build_dataset.py     compile normalized, fixed-length tensors
  train_eval.py        signer-independent + random-split evaluation (metrics only)
  train_model.py       train one deployable model on all data -> model.keras
  live_demo.py         webcam tester using the saved model
  export_tflite.py     model.keras -> model.tflite (+ ood.json) for serving
  parity_spike.py      Holistic vs MediaPipe-Tasks landmarks (mobile feasibility)
  parity_report.py     aggregate parity clips into a go / no-go verdict
  test_client.py       smoke-test a running server without a phone
server/                FastAPI service for the mobile app (see server/README.md)
notebooks/             experiments (import nslr; don't define core logic here)
data/                  raw/ (recorded clips) and processed/ (generated tensors) — created locally, gitignored
results/               metrics.json, model_meta.json (committed) + model.keras, PNGs (generated, gitignored)
requirements.txt       pinned Python dependencies
```

Core logic lives in `nslr/` (importable, diffable, portable to JS later). Scripts
and notebooks stay thin — they import `nslr`.

> **Note on data location:** the code defaults to `data/raw` and
> `data/processed` under the repo root (`nslr/config.py`). Every pipeline
> script also accepts `--data`/`--processed` overrides, so you can instead
> point them at a separately-cloned dataset repo (e.g. a sibling
> `dataset/raw/...` checkout) if your recorded clips live in their own git
> repository. Either layout works as long as the folders match what
> `record.py` produced.

## Prerequisites

- **Python 3.11** recommended. `mediapipe`, `tensorflow`, and `numpy` are
  pinned to versions that are known to work together on 3.11; Python 3.12+
  may fail to resolve some of these pins.
- A **webcam** (for `record.py` and `live_demo.py`). Training/evaluation
  (`build_dataset.py`, `train_eval.py`, `train_model.py`) don't need one.
- Runs fine on **CPU** — TensorFlow has no native-Windows GPU support (≥2.11)
  and the model is small (~192k params). For GPU, use WSL2 or Colab instead.
- No API keys, accounts, or external services are required anywhere in this
  pipeline — all "data collection" is local webcam recording.

## Setup (one environment)

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows  (source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
```

Versions are pinned as a coherent set (mediapipe + TensorFlow in one env forces
an older stack — see the comments in `requirements.txt`).

## Quickstart

If you just want to see the pipeline run end-to-end from scratch:

```bash
# 1. Record a few clips per phrase (GUI opens your webcam)
python scripts/record.py

# 2. Derive SEQ_LEN from what you recorded
python scripts/find_seq_len.py

# 3. Build normalized tensors for training
python scripts/build_dataset.py

# 4. Evaluate (prints accuracy; saves metrics + plots, no model file)
python -u scripts/train_eval.py --verbose 2

# 5. Train a deployable model
python scripts/train_model.py

# 6. Try it live on your webcam
python scripts/live_demo.py
```

You'll need at least a handful of clips per class, ideally from more than one
signer, before steps 4–6 produce meaningful results (see
[Current results](#current-results) for the scale of data that produced
99.7% accuracy: ~130–160 files per class across 4 signers).

## Pipeline — details

Each step reads the previous step's output; re-run the chain end-to-end
whenever new recordings land in `data/raw/`.

### 1. Record clips

```bash
python scripts/record.py
```

Tkinter GUI: pick an output folder and a signer ID, select a phrase, then
**Start/Stop** (or **SPACE**) to record a clip, **Esc** to discard an
in-progress recording. Each clip is saved as:

```
data/raw/<class>/<class>__<signer>__<index>.npy   # (n_frames, 225) landmark array
data/raw/<class>/<class>__<signer>__<index>.json  # metadata: signer, duration, detect rates, ...
```

Your last-used folder/signer is remembered in `.recorder_config.json`
(gitignored, machine-local).

### 2. Find the sequence length

```bash
python scripts/find_seq_len.py                 # add --show to view the histogram
```

Scans clip frame counts and picks `SEQ_LEN` at the 95th percentile (override
with `--percentile`). Writes `data/processed/seq_len.json` and a histogram
PNG. If this step is skipped, later scripts fall back to `SEQ_LEN=151`.

### 3. Build the dataset

```bash
python scripts/build_dataset.py                 # --min-hand-detect 0.5 to drop weak clips
```

Normalizes every clip (dual-anchor: shoulders for pose, wrist/MCP for each
hand) and standardizes it to a fixed length (subsample if longer, zero-pad if
shorter). Writes to `data/processed/`:

- `X.npy` — `(N, seq_len, 225)` float tensor
- `mask.npy` — `(N, seq_len)` boolean padding mask
- `y.npy` — `(N,)` integer labels
- `label_map.json`, `manifest.csv` — bookkeeping
- `dropped.csv` — clips excluded by `--min-hand-detect`, if any

### 4. Train + evaluate

```bash
python -u scripts/train_eval.py --verbose 2
```

(`-u` = unbuffered stdout so progress prints live; `--verbose 2` = one line
per epoch.) Runs two protocols with a fixed seed (42):

1. **Signer-independent (leave-one-signer-out)** — for each signer who has
   all 6 classes recorded, train on everyone else and test on them.
2. **Random-split baseline** — stratified 70/15/15 split.

Model is `nslr.model.build_bilstm()` — a small Bidirectional-LSTM classifier
with masking (for padded frames), dropout, and class-balanced weighting,
trained with early stopping on validation loss. **This script only measures
accuracy — it does not save a model.** Outputs:

- `results/metrics.json` — fold accuracies, confusion matrix, per-class
  precision/recall/F1
- `results/confusion_matrix_signer_indep.png`
- `results/training_curves.png`

### 5. Train a deployable model

```bash
python scripts/train_model.py
```

Trains one BiLSTM on **all** processed data (85/15 train/val split) and
saves:

- `results/model.keras` — the trained model
- `results/model_meta.json` — `class_names`, `seq_len`, `confidence_threshold`

### 6. Try it live

```bash
python scripts/train_model.py                   # -> results/model.keras + model_meta.json
python scripts/live_demo.py                      # opens the webcam
```

In the tester: **SPACE** to start a sign, **SPACE** again to stop and classify,
**R** to clear, **Q**/**Esc** to quit. It's segment-based (record a whole sign,
then predict) because the model expects one complete sign per input and has no
"idle" class — continuous always-on prediction would be unreliable. A confidence
threshold (default 0.75) suppresses unsure guesses.

### 7. Serve it to a mobile app

```bash
python scripts/export_tflite.py                  # -> results/model.tflite (882 KB)
pip install fastapi "uvicorn[standard]"
uvicorn server.app:app --host 0.0.0.0 --port 8000
python scripts/test_client.py                    # verify without a phone
```

A Flutter client streams JPEG frames over `WS /ws/stream` while the user signs;
the server runs the same Holistic → `nslr.preprocess` → BiLSTM path as
`live_demo.py` and returns the phrase. Full protocol and deployment notes in
[server/README.md](server/README.md).

## Current results

From the committed `results/metrics.json` (6 classes, seq_len=151, 4 signers,
~130–160 clips per class):

| Protocol | Accuracy |
|---|---|
| Signer-independent (mean over 3 held-out signers) | 99.74% |
| Signer-independent (pooled) | 99.73% |
| Random-split baseline | 98.59% |

All per-class F1 scores are ≥0.99. Re-running `train_eval.py` on your own
recordings will overwrite this file with your own numbers.

## Repo housekeeping notes

- **No `LICENSE` or `CONTRIBUTING.md`** exists yet in this repo — add one if
  you plan to open it up beyond the project team.
- **`main.pdf`** (the design proposal referenced in code comments as
  "proposal 3.3", "4.3.5", etc.) is not part of this repo; treat those
  comments as pointers to an external document, not a local file.
- `data/`, `results/*.png`, `results/*.keras`, and `.recorder_config.json`
  are all gitignored — expect to regenerate them locally rather than finding
  them after a fresh clone.
- `notebooks/` currently only contains a `README.md` with conventions
  (import `nslr`, keep notebooks thin) — no notebooks are committed yet.
- Steps 2–4 above only regenerate `data/processed/` and `results/`; they
  never touch `data/raw/`. Delete those two folders to reset from scratch.
