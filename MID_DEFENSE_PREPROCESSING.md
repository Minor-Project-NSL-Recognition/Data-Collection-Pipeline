# Mid-Defense Study Guide — Data Collection & Preprocessing

This document covers **everything up to and including data preprocessing** for the
NSL emergency-phrase recognition project: how raw sign clips are captured, how a
sequence length is chosen, and how clips are turned into the fixed-length,
normalized tensors a model would later train on. The model, training, open-set
rejection, and live inference stages are **deliberately out of scope** for this
checkpoint and have been removed from this branch (`mid-defense-preprocessing`)
so the repo reads as exactly the work being defended.

Read this top to bottom once; the **[Likely defense questions](#9-likely-defense-questions-cheat-sheet)**
section at the end is the fast revision pass.

---

## 1. The 30-second summary

We recognise six Nepali Sign Language (NSL) emergency phrases (plus a `none`
"not-a-sign" class) from a webcam. Instead of feeding **raw video pixels** to a
model, we first convert every frame into a compact set of **body and hand
landmark coordinates** using Google MediaPipe Holistic. A sign therefore becomes
a short *time-series of landmark vectors*. The preprocessing stage's job is to
turn those messy, variable-length recordings into a clean, uniform numeric
tensor:

```
webcam video  →  landmarks per frame  →  raw clip (variable length)
             →  normalized clip  →  fixed-length clip  →  X / y tensors
```

Everything in this document is that arrow chain.

---

## 2. Why landmark-based, not pixels? (the single most important design choice)

A naïve approach feeds raw camera frames (e.g. 640×480×3 ≈ 920k numbers/frame)
into a CNN. We instead extract **225 numbers/frame**. Why:

| Concern | Raw pixels | Landmarks (our choice) |
|---|---|---|
| Input size | ~920,000 / frame | **225 / frame** (~4000× smaller) |
| Background, clothing, lighting | Model must learn to ignore them | **Gone** — landmarks are geometry only |
| Data needed to generalise | Huge | Small (we succeed with ~570 clips) |
| Signer appearance / skin tone / camera | Confounds the model | Irrelevant — only joint positions remain |
| Portable to the browser later | Heavy | MediaPipe runs client-side; same 225 contract |

**The core idea:** a sign is defined by *how the body and hands move*, not by what
the signer looks like or what's behind them. Landmarks throw away everything
except that movement, so the model can learn the gesture from far less data. This
is the reasoning to lead with in the defense.

**Trade-off to acknowledge:** we are fully dependent on MediaPipe's detector. If
it fails to find a hand (occlusion, motion blur, out of frame), that information
is simply missing for that frame. We handle this explicitly (zero-filling +
quality metrics — see §4.3), which is why detection-rate tracking exists.

---

## 3. The feature contract — `nslr/config.py`

Everything downstream agrees on one fixed layout, defined once in
[`config.py`](nslr/config.py) so the recorder, the preprocessor, and (later) a
JavaScript browser port can all be checked against a single source of truth.

### 3.1 The 225-vector

Each frame is a flat vector of **225 floats**:

```
[  pose: 33 landmarks × 3  |  left hand: 21 × 3  |  right hand: 21 × 3  ]
        = 99                        = 63                    = 63          = 225
```

- **33 pose landmarks** — body skeleton (shoulders, elbows, wrists, face anchors…)
- **21 hand landmarks each** — MediaPipe's hand model gives 21 points per hand
  (wrist + 4 joints × 5 fingers)
- **× 3** — each landmark is `(x, y, z)`. `x, y` are normalized image
  coordinates in `[0, 1]`; `z` is depth relative to the hips/wrist. We keep `z`
  because sign language uses motion toward/away from the body.

`FEATURE_DIM = POSE_DIM + HAND_DIM*2 = 99 + 63 + 63 = 225`. The three fixed
**slices** (`POSE_SLICE`, `LEFT_HAND_SLICE`, `RIGHT_HAND_SLICE`) let any later
code address one body part without magic numbers.

### 3.2 Why pose *and* both hands (Holistic, not just Hands)

Emergency phrases involve the hands relative to the **body** — e.g. a hand moving
to the chest ("can't breathe") vs. to the side. Hand landmarks alone are
normalized to the hand itself and lose that "where on the body" context. Pose
landmarks (especially the shoulders) give us that anchor. That's why we use
MediaPipe **Holistic** (pose + face + both hands) rather than the Hands solution
alone.

### 3.3 The normalization anchors (defined here, applied in §6)

`config.py` also names the specific landmark indices used later for
normalization, so the "geometry" is locked in one place:

- Pose: `POSE_LEFT_SHOULDER = 11`, `POSE_RIGHT_SHOULDER = 12`
- Hand: `HAND_WRIST = 0`, `HAND_MIDDLE_FINGER_MCP = 9`
- `EPS = 1e-6` — added to every scale denominator so we never divide by zero.

### 3.4 The class list

`CLASSES` maps folder-safe keys to GUI labels for the six phrases plus a seventh
`none` negatives class. Phrase #6 is `need_toilet` by design (the original
written proposal says "earthquake"; **the recorded data is the source of truth**
— a good thing to state plainly if asked about the discrepancy).

---

## 4. Stage 1 — Data collection (`scripts/record.py` + `nslr/landmarks.py`)

**Goal:** capture raw, variable-length landmark clips from a webcam, one file per
sign performance, with quality metadata.

### 4.1 The recorder

[`record.py`](scripts/record.py) is a Tkinter GUI. The operator picks an **output
folder** and a **signer ID**, selects a **phrase**, and presses **Start/Stop**
(or SPACE) to bracket one performance of the sign. Design points worth defending:

- **Manual start/stop segmentation.** The human decides exactly when a sign
  begins and ends. This gives clean, well-bracketed clips and sidesteps the much
  harder "automatic gesture spotting" problem — appropriate for a data-collection
  tool where quality matters more than automation.
- **Mirror flip before detection** (`cv2.flip(frame, 1)`). The preview is
  mirrored so it feels like a mirror to the signer (natural). Crucially, the
  landmarks are extracted from the *already-flipped* frame, so "left hand" in our
  data means the signer's left as they see it. **Any future inference must apply
  the identical flip** or the hands swap — this consistency is the reason the
  flip lives at capture time.
- **Minimum length guard.** Clips under 5 frames are rejected as accidental
  taps.

### 4.2 Landmark extraction — the MediaPipe → 225 bridge

Every recorded frame passes through
[`extract_frame_vector`](nslr/landmarks.py) which turns a MediaPipe Holistic
result into the 225-vector:

```python
pose  = [ (lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark ]  # 99
left  = ... left_hand_landmarks ...                                        # 63
right = ... right_hand_landmarks ...                                       # 63
vector = concatenate([pose, left, right])   # 225, asserted
```

### 4.3 Missing detections → zero-fill + flags (important)

If MediaPipe fails to find, say, the left hand in a frame, that block is filled
with **zeros** and a per-frame flag records the miss. Two reasons this matters:

1. **Zeros are a deliberate, maskable sentinel.** Because we normalize *before*
   padding (§6.3) and the normalization maps zero → zero, missing/padded values
   stay exactly `0.0` and can later be ignored by a masking layer. Consistency of
   "missing = 0" is a design invariant, not an accident.
2. **Quality metrics.** Per clip we compute `pose_detect_rate`,
   `left/right_hand_detect_rate`, and `any_hand_detect_rate` (fraction of frames
   in which at least one hand was found) and save them alongside the clip. These
   drive an optional quality gate later (§6.4) and warn the operator to
   re-record if hands were rarely seen (< 70%).

### 4.4 What gets saved, and the naming convention

Per performance, two files land in `data/raw/<class>/`:

```
<class>__<signer>__<index>.npy    # float32 array, shape (n_frames, 225)
<class>__<signer>__<index>.json   # metadata: signer, duration, detect rates, ...
```

The `<class>__<signer>__<index>` stem is not cosmetic — the embedded **signer ID
is later parsed back out** (§6.5) so we can evaluate the model in a
*signer-independent* way (train on some people, test on people the model has
never seen). Recording multiple signers is what makes that possible.

### 4.5 Why store RAW and variable-length (not normalized here)

The recorder does **no** normalization or length-fixing. That is a deliberate
separation of concerns:

- Normalization scheme and `SEQ_LEN` become **cheap, re-runnable knobs** — we can
  change the preprocessing and rebuild tensors in seconds without ever
  re-recording a human.
- The raw capture is the ground truth; everything else is reproducible from it.

### 4.6 What we actually collected (current numbers)

| Metric | Value |
|---|---|
| Total clips | **570** |
| Classes | **7** (6 phrases + `none`) |
| Signers | **4** (`signer01`–`signer04`), ~140 clips each |
| Clips per (signer × class) | ~20 (very balanced) |
| Frames per clip | min **33**, mean **92**, median **90**, max **217** |
| `any_hand_detect_rate` | mean **0.78** |

The balance (≈20 clips per signer per class) is worth highlighting: no class or
signer dominates, so the dataset itself introduces little bias.

---

## 5. Stage 2 — Choosing the sequence length (`scripts/find_seq_len.py`)

**Problem:** clips range from 33 to 217 frames, but a batched recurrent model
needs a **single fixed length** `SEQ_LEN` for every sample. What length?

### 5.1 The method

[`find_seq_len.py`](scripts/find_seq_len.py) reads the frame count of every raw
clip and takes a **percentile** of that distribution (default **95th**):

```
SEQ_LEN = round( percentile(frame_counts, 95) )
```

For our data the 95th percentile is **148**, so `SEQ_LEN = 148`. It writes
`data/processed/seq_len.json` (consumed by the next stage) plus a histogram PNG.

### 5.2 Why the 95th percentile — the trade-off

`SEQ_LEN` sets a single cutoff, and every clip is forced to it:

- Clips **longer** than `SEQ_LEN` are **subsampled** (some frames dropped).
- Clips **shorter** than `SEQ_LEN` are **zero-padded** (padding added).

So the choice is a tension:

- **Too short** (e.g. the median, 90) → the longest ~half of clips get
  aggressively subsampled and may lose detail.
- **Too long** (e.g. the max, 217) → almost every clip is mostly padding, which
  is wasteful and dilutes the real signal.

The **95th percentile** is the standard compromise: ~95% of clips are captured
(near-)complete with at most mild subsampling, while the rare very long clips
(the top 5%, up to 217 frames) absorb the subsampling instead of everyone paying
for them with padding. It's a data-driven cutoff, not a guessed constant.

### 5.3 Robustness

If this step is skipped, downstream code falls back to `FALLBACK_SEQ_LEN = 151`.
The value is a knob (`--percentile`), so the whole tensor set can be rebuilt with
a different length in one command — again thanks to keeping raw data untouched.

---

## 6. Stage 3 — Preprocessing into tensors (`nslr/preprocess.py` + `nslr/dataset.py`)

This is the heart of the defense. `build_dataset.py` drives it; the real logic is
in `preprocess.py` (per-clip math) and `dataset.py` (assembling the dataset).
Two transforms happen per clip: **normalize**, then **standardize length**.

### 6.1 Normalization — dual-anchor, and *why*

MediaPipe gives coordinates in image space, so the same sign looks numerically
different if the signer stands closer, further, left, or right. The model should
learn the **gesture shape**, not the signer's position in frame. We remove
position and scale with a **dual-anchor** scheme — one anchor for the body, a
separate one for each hand.

**Pose block** — anchor on the shoulders:

```
mid          = ½ (left_shoulder + right_shoulder)      # origin
shoulder_w   = ‖ left_shoulder − right_shoulder ‖       # scale
p_normalized = (p − mid) / (shoulder_w + ε)
```

This makes pose **translation-invariant** (subtract the body centre) and
**scale-invariant** (divide by shoulder width — a stable body measurement that
barely changes within a person and is comparable across people). Standing nearer
or further no longer changes the numbers.

**Each hand block** — anchor on that hand's own wrist:

```
scale        = ‖ middle_finger_MCP − wrist ‖            # hand size
h_normalized = (h − wrist) / (scale + ε)
```

Each hand is normalized **relative to itself** (wrist as origin, wrist→knuckle
distance as scale), so finger shape is captured independently of where the hand
is or how big it appears.

**Why two different anchors (the key subtlety):** the pose anchor keeps the
*"where is the hand relative to the body"* information (hand-to-chest vs.
hand-to-side survives because pose is normalized in body coordinates), while the
per-hand anchor cleanly captures *finger configuration* regardless of hand
position. One global anchor could not do both well. `ε = 1e-6` guards every
division.

### 6.2 A crucial ordering: normalize **before** padding

The blocks are normalized independently, and the anchor formulas all map an
all-zero input to an all-zero output. Therefore a missing landmark block (§4.3)
**stays exactly zero after normalization**, and so does any padding we add next.
That is what makes `0.0` a reliable "ignore me" sentinel that a masking layer can
later skip. If we padded first and normalized after, padding could become nonzero
and pollute the signal. (Stated in the module docstring; a likely probe question.)

### 6.3 Standardizing length — subsample or pad

[`standardize_length`](nslr/preprocess.py) forces each normalized clip to exactly
`SEQ_LEN` frames and returns a boolean **mask** marking real vs. padded frames:

| Case | Action | Mask |
|---|---|---|
| `n == SEQ_LEN` | keep as-is | all `True` |
| `n  > SEQ_LEN` | **uniform subsample**: `np.linspace(0, n−1, SEQ_LEN)` indices | all `True` |
| `n  < SEQ_LEN` | **zero-pad at the end** | `True` for real frames, `False` for padding |

**Why *uniform* subsampling** (evenly spaced across the whole clip) rather than
truncating the tail: it preserves the **entire gesture arc** — beginning, middle,
and end — just at a slightly coarser time resolution. Chopping the end off would
throw away the sign's conclusion. Padding goes **at the end** so all the real
motion stays contiguous from frame 0, which is exactly what a masking layer
expects.

### 6.4 Assembling the dataset — `compile_dataset`

[`compile_dataset`](nslr/dataset.py) walks `data/raw/`, applies the two
transforms to every clip, and stacks the results. Notable robustness logic:

- **Skip empty class folders.** Only folders that actually contain `.npy` files
  become labels, so a not-yet-recorded class can't create a phantom zero-sample
  label. (This is exactly how `none` was introduced gradually.)
- **Optional quality gate** (`--min-hand-detect`). Clips whose
  `any_hand_detect_rate` is below a threshold can be dropped, logged with the
  reason to `dropped.csv`. Off by default (keeps all clips).
- **Shape validation.** Any clip that isn't `(n, 225)` is dropped and logged
  rather than crashing the build.
- **Bookkeeping.** A `manifest.csv` records, per sample, its class, **signer**,
  source file, original frame count, and which standardization branch it hit
  (exact / subsampled / padded).

### 6.5 The outputs (what "preprocessing done" produces)

Written to `data/processed/`:

| File | Shape / content | Purpose |
|---|---|---|
| `X.npy` | `(N, SEQ_LEN, 225)` float32 | the model input tensor |
| `mask.npy` | `(N, SEQ_LEN)` bool | which frames are real vs. padding |
| `y.npy` | `(N,)` int64 | integer class label per sample |
| `label_map.json` | name → index | decode predictions back to phrases |
| `manifest.csv` | per-sample provenance incl. **signer** | traceability + signer-independent splits |
| `dropped.csv` | excluded clips + reason | data-quality audit trail |

`build_dataset.py` also prints a summary: total shape, the exact/subsampled/padded
counts, overall **padding fraction** (`1 − mask.mean()`), per-class counts, and
per-signer counts — a quick sanity check that the build is balanced.

### 6.6 Where preprocessing ends (the handoff)

`dataset.py` also has `load_processed` (reads the tensors back, re-attaching the
signer column from the manifest) and `eligible_test_signers` (which signers can
be held out such that every class still appears in training). These are the
**boundary** to the training stage — included here only to show *why* we tracked
signer IDs all the way from capture: so a later stage can measure
signer-independent accuracy. Training itself is out of scope for this defense.

---

## 7. End-to-end: how to run this stage

```bash
# 1. Record clips (webcam GUI; pick folder, signer, phrase; SPACE to record)
python scripts/record.py

# 2. Derive SEQ_LEN from the recorded frame-count distribution
python scripts/find_seq_len.py            # add --show to view the histogram

# 3. Build normalized, fixed-length tensors
python scripts/build_dataset.py           # --min-hand-detect 0.5 to drop weak clips
```

After step 3, `data/processed/` holds `X.npy`, `mask.npy`, `y.npy`, and the
bookkeeping files — the complete output of the preprocessing stage.

> Only `record.py` needs a webcam. `find_seq_len.py` and `build_dataset.py` need
> only NumPy/Matplotlib — no TensorFlow — so the preprocessing stage is light to
> run and reason about on its own.

---

## 8. Design principles running through the whole stage

1. **Separate capture from processing.** Raw clips are immutable ground truth;
   normalization and `SEQ_LEN` are cheap knobs re-applied without re-recording.
2. **One source of truth for the data contract.** `config.py` fixes the
   225-layout, anchors, and class list so recorder, preprocessor, and a future
   JS port can't drift apart.
3. **Invariance by construction.** Landmarks remove appearance/background;
   dual-anchor normalization removes position/scale. The model only ever sees
   gesture geometry.
4. **`0.0` means "ignore".** Missing detections and padding are both exactly
   zero *by design*, made safe by normalizing before padding, so a masking layer
   can skip them.
5. **Traceability.** Every sample carries its signer and provenance, enabling
   honest signer-independent evaluation downstream.

---

## 9. Likely defense questions (cheat sheet)

**Q. Why landmarks instead of feeding the video to a CNN?**
A. A sign is defined by motion geometry, not appearance. Landmarks cut the input
from ~920k to 225 numbers/frame, remove background/lighting/signer-appearance
confounds, and let us generalise from only ~570 clips. Trade-off: we depend on
MediaPipe's detector, so we track detection rates and handle misses explicitly.

**Q. Why 225 features?**
A. 33 pose + 21 left-hand + 21 right-hand landmarks, each `(x, y, z)`:
33·3 + 21·3 + 21·3 = 99 + 63 + 63 = 225.

**Q. Why keep the `z` coordinate?**
A. Signs use depth — motion toward/away from the body — so we keep all three
axes, not just the 2D image position.

**Q. Why Holistic (pose + hands), not just the hand detector?**
A. We need the hands *relative to the body* (hand-to-chest vs. hand-to-side).
Hand-only landmarks lose that; the shoulders give the body anchor.

**Q. Why normalize, and why two different anchors?**
A. To make the sign invariant to where the signer stands and how big they appear.
The **shoulder** anchor puts pose in body-centred, shoulder-width units (keeps
hand-relative-to-body info); each **hand** is normalized to its own wrist and
knuckle span (captures finger shape independent of hand position). One anchor
can't do both.

**Q. Why normalize *before* padding?**
A. The normalization maps zero to zero, so missing blocks and padding stay
exactly `0.0` and remain maskable. Normalizing after padding could make padded
frames nonzero and corrupt the signal.

**Q. How do you pick SEQ_LEN, and why the 95th percentile?**
A. From the actual frame-count distribution. The 95th percentile (=148 here)
keeps ~95% of clips essentially complete while letting the rare very long clips
(up to 217) absorb subsampling, instead of forcing everyone to pad up to the max.

**Q. What happens to clips that aren't exactly SEQ_LEN?**
A. Longer → uniformly subsampled across the whole clip (keeps the full gesture
arc, coarser time step). Shorter → zero-padded at the end, with a boolean mask
marking the padding.

**Q. Why uniform subsampling and not just truncating?**
A. Truncating deletes the end of the sign. Uniform (evenly spaced) sampling keeps
the beginning, middle, and end.

**Q. What does the mask do?**
A. It marks real vs. padded frames so a downstream masking layer ignores padding
and doesn't treat zeros as real motion.

**Q. How do you handle a frame where MediaPipe misses a hand?**
A. That block is zero-filled and a per-frame flag is recorded; per-clip detection
rates are stored and can gate out low-quality clips (`--min-hand-detect`).

**Q. Why record multiple signers, and why encode signer in the filename?**
A. So we can later train on some people and test on unseen people
(signer-independent evaluation). The signer is parsed from the filename into
`manifest.csv` and carried with every sample.

**Q. What is the `none` class for?**
A. Negatives — rest, random motion, not-a-sign — so the system can later learn to
reject non-signs instead of forcing one of the six labels. During preprocessing
it's just a seventh class folder; empty-folder handling let it be added
gradually.

**Q. How much padding is in the final tensors?**
A. `build_dataset.py` prints `1 − mask.mean()`; with SEQ_LEN=148 and a median
clip of ~90 frames, expect a moderate padding fraction — the mask makes it
harmless.

**Q. Why is phrase #6 `need_toilet` when the proposal says "earthquake"?**
A. The recorded dataset is the source of truth; the phrase set was adjusted
during collection and the code/data reflect the real signs recorded.

---

## 10. Scope note for this branch

This branch (`mid-defense-preprocessing`) intentionally contains **only** the
files for capture through preprocessing:

```
nslr/config.py        landmarks.py   preprocess.py   dataset.py   __init__.py
scripts/record.py     find_seq_len.py   build_dataset.py
```

The model (`model.py`), training (`train_eval.py`, `train_model.py`), open-set
rejection (`ood.py`), live inference (`live_demo.py`), and their result artifacts
were removed relative to `master` so the branch matches exactly what this
checkpoint defends. They remain on `master` for the later defense.
