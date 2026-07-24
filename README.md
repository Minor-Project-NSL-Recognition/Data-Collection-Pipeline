# NSL Emergency-Phrase Recognition — Data & Preprocessing Pipeline

> **Branch scope — `mid-defense-preprocessing`.** This branch contains **only the
> data collection and preprocessing stages** (webcam → landmarks → normalized,
> fixed-length tensors), for the mid-defense checkpoint. The model, training,
> open-set rejection, and live inference stages live on `master`. For a detailed
> walkthrough with the reasoning behind every step, see
> **[MID_DEFENSE_PREPROCESSING.md](MID_DEFENSE_PREPROCESSING.md)**.

Landmark-based recognition of six Nepali Sign Language (NSL) emergency phrases
(plus a `none` negatives class): MediaPipe Holistic → dual-anchor normalization →
fixed-length tensors ready for a sequence model. This repo is the offline data
side of a larger project; `main.pdf` (the full proposal) is kept outside this
repo, so code comments referencing "proposal X.Y" won't resolve to a local file.

## The six phrases

| Key | GUI label |
|---|---|
| `cant_breathe` | 1. I can't breathe (Medical) |
| `building_on_fire` | 2. The building is on fire (Fire) |
| `call_police` | 3. Call the police (Crime) |
| `need_ambulance` | 4. I need an ambulance (Medical) |
| `help_danger` | 5. Help me / I am in danger (Generic) |
| `need_toilet` | 6. I need to go to the toilet (Basic need) |

Plus `none` — negatives (rest, random motion, not-a-sign) for later open-set
rejection. (Phrase #6 is `need_toilet` by design — the proposal text still says
"earthquake"; the data/code here is the source of truth.)

## Layout (this branch)

```
nslr/                  importable package — single source of truth
  config.py            locked constants, class list, paths, seq_len fallback
  landmarks.py         MediaPipe result -> 225-vector
  preprocess.py        dual-anchor normalization + length standardization
  dataset.py           raw clips -> X/mask/y ; load back + signer-split helpers
scripts/               thin entrypoints (import nslr)
  record.py            GUI landmark recorder (webcam -> raw clips)
  find_seq_len.py      derive SEQ_LEN from the recorded frame-count distribution
  build_dataset.py     compile normalized, fixed-length tensors
notebooks/             experiments (import nslr; don't define core logic here)
data/                  raw/ (recorded clips) and processed/ (generated tensors) — created locally, gitignored
requirements.txt       pinned Python dependencies
```

Core logic lives in `nslr/` (importable, diffable, portable to JS later). Scripts
and notebooks stay thin — they import `nslr`.

> **Note on data location:** the code defaults to `data/raw` and
> `data/processed` under the repo root (`nslr/config.py`). Every pipeline
> script also accepts `--data`/`--out` overrides, so you can instead point them
> at a separately-cloned dataset repo if your recorded clips live in their own
> git repository. Either layout works as long as the folders match what
> `record.py` produced.

## Prerequisites

- **Python 3.11** recommended. `mediapipe` and `numpy` are pinned to versions
  known to work together on 3.11; Python 3.12+ may fail to resolve some pins.
- A **webcam** (for `record.py` only). `find_seq_len.py` and `build_dataset.py`
  don't need one — and don't need TensorFlow either, just NumPy/Matplotlib.
- Runs fine on **CPU**.
- No API keys, accounts, or external services anywhere — all "data collection" is
  local webcam recording.

## Setup (one environment)

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows  (source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
```

## Quickstart — the preprocessing pipeline

```bash
# 1. Record a few clips per phrase (GUI opens your webcam)
python scripts/record.py

# 2. Derive SEQ_LEN from what you recorded
python scripts/find_seq_len.py

# 3. Build normalized tensors (the output of this stage)
python scripts/build_dataset.py
```

You'll want at least a handful of clips per class, ideally from more than one
signer. The committed dataset that this checkpoint is built on is **570 clips**
across **7 classes** and **4 signers** (~20 clips per signer per class).

## Pipeline — details

Each step reads the previous step's output; re-run the chain end-to-end whenever
new recordings land in `data/raw/`. **See
[MID_DEFENSE_PREPROCESSING.md](MID_DEFENSE_PREPROCESSING.md) for the full
reasoning behind each step** — the summary below is just the commands.

### 1. Record clips

```bash
python scripts/record.py
```

Tkinter GUI: pick an output folder and a signer ID, select a phrase, then
**Start/Stop** (or **SPACE**) to record a clip, **Esc** to discard an in-progress
recording. Each clip is saved as:

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

Scans clip frame counts and picks `SEQ_LEN` at the 95th percentile (override with
`--percentile`). Writes `data/processed/seq_len.json` and a histogram PNG. If
skipped, later scripts fall back to `SEQ_LEN=151`.

### 3. Build the dataset

```bash
python scripts/build_dataset.py                 # --min-hand-detect 0.5 to drop weak clips
```

Normalizes every clip (dual-anchor: shoulders for pose, wrist/MCP for each hand)
and standardizes it to a fixed length (subsample if longer, zero-pad if shorter).
Writes to `data/processed/`:

- `X.npy` — `(N, seq_len, 225)` float tensor
- `mask.npy` — `(N, seq_len)` boolean padding mask
- `y.npy` — `(N,)` integer labels
- `label_map.json`, `manifest.csv` — bookkeeping (incl. signer per sample)
- `dropped.csv` — clips excluded by `--min-hand-detect`, if any

That completes the preprocessing stage: `data/processed/` now holds model-ready
tensors.

## Repo housekeeping notes

- **`main.pdf`** (the design proposal referenced in code comments as
  "proposal 3.3", "4.3.5", etc.) is not part of this repo; treat those comments
  as pointers to an external document, not a local file.
- `data/` and `.recorder_config.json` are gitignored — expect to regenerate
  `data/processed/` locally rather than finding it after a fresh clone.
- Steps 2–3 only regenerate `data/processed/`; they never touch `data/raw/`.
  Delete `data/processed/` to rebuild from scratch.
- The model, training, evaluation, open-set rejection, and live-demo code are on
  `master`, not on this preprocessing-scope branch.
