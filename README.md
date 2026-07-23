# NSL Emergency-Phrase Recognition — Data & Model Pipeline

Landmark-based recognition of six Nepali Sign Language emergency phrases
(MediaPipe Holistic → dual-anchor normalization → BiLSTM). This repo is the
offline data + training side; see `main.pdf` for the full proposal.

## Layout

```
nslr/                 importable package — single source of truth
  config.py           locked constants, class list, paths, seq_len fallback
  landmarks.py        MediaPipe result -> 225-vector
  preprocess.py       dual-anchor normalization + length standardization
  dataset.py          raw clips -> X/mask/y ; load back for training
  model.py            the BiLSTM
scripts/              thin entrypoints (import nslr)
  record.py           GUI landmark recorder
  find_seq_len.py     derive SEQ_LEN from the data
  build_dataset.py    compile processed tensors
  train_eval.py       train + signer-independent evaluation
notebooks/            experiments (import nslr; don't define core logic here)
data/                 raw/ (recorded clips) and processed/ (generated tensors)
results/              metrics.json + plots from train_eval
```

Core logic lives in `nslr/` (importable, diffable, portable to JS later). Scripts
and notebooks stay thin — they import `nslr`.

## Setup (one environment)

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows  (source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
```

Versions are pinned as a coherent set (mediapipe + TensorFlow in one env forces an
older stack — see the note in `requirements.txt`). Training runs on **CPU**:
TensorFlow has no native-Windows GPU support (≥2.11), and the model is small
enough that CPU is fine. For GPU later, use WSL2 or Colab.

## Pipeline — run after adding clips to `data/raw/`

Each step reads the previous step's output; re-run the chain end-to-end whenever
new recordings land in `data/raw/`.

```bash
# 1. Record clips (GUI). Point it at this repo's data/ folder.
python scripts/record.py

# 2. Derive SEQ_LEN from the current frame-count distribution.
#    Writes data/processed/seq_len.json + a histogram.
python scripts/find_seq_len.py                 # add --show to view the histogram

# 3. Compile normalized, fixed-length tensors -> data/processed/{X,mask,y}.npy
python scripts/build_dataset.py                 # --min-hand-detect 0.5 to drop weak clips

# 4. Train + evaluate (signer-independent + random-split baseline).
#    -u = unbuffered so progress prints live; --verbose 2 = one line per epoch.
python -u scripts/train_eval.py --verbose 2
```

Outputs:
- `data/processed/` — `X.npy` `(N, seq_len, 225)`, `mask.npy`, `y.npy`, `label_map.json`, `manifest.csv`, `seq_len.json`
- `results/` — `metrics.json`, `confusion_matrix_signer_indep.png`, `training_curves.png`

## Test the trained model live (webcam)

`train_eval.py` only measures accuracy — it saves no model. To try the model on
a webcam, train a deployable one on all the data, then run the tester:

```bash
python scripts/train_model.py                   # -> results/model.keras + model_meta.json
python scripts/live_demo.py                      # opens the webcam
```

In the tester: **SPACE** to start a sign, **SPACE** again to stop and classify,
**R** to clear, **Q**/**Esc** to quit. It's segment-based (record a whole sign,
then predict) because the model expects one complete sign per input and has no
"idle" class — continuous always-on prediction would be unreliable.

### Rejecting unknown / wrong signs (open-set)

A 6-class softmax is closed-world: it always picks one of the 6, confidently,
even for a wrong or mixed sign. Two defenses:

1. **Distance gate (built in).** `train_model.py` fits a prototype per class in
   the model's embedding space and stores it in `results/ood_stats.npz`. The demo
   rejects an input whose Mahalanobis distance to every known sign exceeds a
   threshold, showing *"Unknown sign — rejected"* regardless of the softmax score.
2. **A real `none` class (fuller fix, needs recording).** Record rest / random /
   partial / mixed gestures into the `none` class (it appears in `record.py`), then
   re-run `build_dataset.py` + `train_model.py`. The pipeline picks up `none`
   automatically once it has clips and trains it as a 7th "not a sign" class.

Neither fully eliminates confident-wrong on out-of-distribution input — that is a
fundamental property of neural classifiers — but together they cut it sharply.

## Notes

- **Phrases (6):** `cant_breathe`, `building_on_fire`, `call_police`, `need_ambulance`,
  `help_danger`, `need_toilet`. Phrase #6 is `need_toilet` by design — the proposal
  text still says "earthquake" and is the thing that's out of date.
- Steps 2–4 only regenerate `data/processed/` and `results/`; they never touch
  `data/raw/`. Delete those two folders to reset.
