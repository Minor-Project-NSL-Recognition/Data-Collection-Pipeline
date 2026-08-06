# Mid-Defense Presentation Script

A slide-by-slide speaking guide for **"NSL Recognition for Emergency Phrases"**
(mid-term defense, 24 July 2026). Each slide has a **Say** block — natural spoken
lines, not the bullets read aloud — plus **Cue** (what to point at) and **If asked**
(quick answers) where useful.

> Golden rule: **talk to the idea, don't read the slide.** The audience can read
> the bullets; your job is to explain them in plain words and sound like you built
> it (you did).

---

## Before you start — the essentials

**The one sentence to memorize** (your whole project in a breath):
> *"We built and tested the complete data-collection and preprocessing pipeline
> that turns webcam sign-language videos into clean, model-ready data — 570 clips
> of six NSL emergency phrases plus a 'none' class, across four signers."*

**What this mid-term is really about:** the **data pipeline is done**. The model is
the *next* phase. If an examiner asks "where are the recognition results / the
BiLSTM?", answer confidently: *"Model training is our next phase — this checkpoint
delivers the dataset and preprocessing pipeline, which is complete and tested."*
Don't apologize for it; a solid data foundation is the point of this stage.

**Two things to be proud of — lean on these:**
1. **Real Deaf-community field work** — you visited actual organizations (RCRD & CBR
   Bhaktapur, NFDN) instead of inventing signs.
2. **A clean, signer-labeled dataset** — because every clip is tagged by signer, you
   can later test on *unseen* people (signer-independent), which most prior NSL work
   did **not** do.

**Timing:** aim for ~8–10 minutes. ~30 seconds per content slide; move faster
through the per-paper literature slides. Leave time for questions.

**Hand-offs:** end your section with *"I'll now hand over to [name]."* Keep it smooth.

**For questions:** the detailed answer sheet is in
[MID_DEFENSE_PREPROCESSING.md](MID_DEFENSE_PREPROCESSING.md), §9. Skim it beforehand.

---

## Suggested presenter split (adjust freely)

Four presenters — Asul, Dipesh, Manee, Sajan. One balanced way to divide it:

| Presenter | Slides | Section |
|---|---|---|
| **Presenter 1** | 1–3 | Title, Outline, Introduction |
| **Presenter 2** | 4–11 | Motivation, Scope, Literature Review |
| **Presenter 3** | 12–16 | Methodology: field visits, recorder, outcomes |
| **Presenter 4** | 17–20 | Pipeline diagram, Result, Conclusion |

The Literature Review is content-heavy (slides 6–11) — Presenter 2 may want to share
it with a teammate, or present the table (slide 6) in full and keep slides 7–11 to
one line each. **Whoever presents the pipeline diagram (slide 17) and Result (slide
18) should be the most comfortable with the technical detail** — those are the two
slides examiners probe.

---

## Slide-by-slide script

### Slide 1 — Title
**Say:** "Good [morning/afternoon]. We're presenting our minor project — *NSL
Recognition for Emergency Phrases*. The goal is a system that reads Nepali Sign
Language from an ordinary webcam and recognizes six emergency phrases in real time,
using MediaPipe for tracking and a BiLSTM sequence model. I'm [name], presenting with
[names]."
**Cue:** Keep it to ~15 seconds — don't linger on the title.

### Slide 2 — Presentation Outline
**Say:** "Here's our flow: we'll cover the introduction and motivation, define our
scope, review related work, and then spend most of our time on the methodology and
results — which is where this mid-term stands — before the conclusion and next steps."
**Cue:** ~10 seconds. Don't read all seven items; just signpost.

### Slide 3 — Introduction
**Say:** "Sign language is how the Deaf community communicates. But in an emergency,
there's usually no interpreter nearby, and every second matters. That's the gap we're
targeting — six critical emergency phrases. Instead of any special hardware, we use
just a webcam that reads the signer's hands and body. And we focus on Nepali Sign
Language specifically, because it has very little existing technology or data."

### Slide 4 — Motivation & Objective
**Say:** "Our motivation is that Deaf people face real barriers in emergencies, and
Nepali Sign Language is badly under-resourced — almost no datasets or tools exist. So
we set four objectives: build a reliable pipeline that turns sign videos into clean
landmark data; collect a dataset across *multiple* signers so it generalizes; prepare
that data for a sequence model; and keep the whole design light enough to run in a
browser later."

### Slide 5 — Scope
**Say:** "To stay focused: we handle six fixed emergency phrases, plus a seventh
'none' class for anything that isn't one of them. Each clip is one sign. A key choice
— we store *landmarks*, the coordinate points, not the raw video. That keeps the data
tiny and protects privacy. It runs on a normal CPU with just a webcam. And for this
mid-term, our current phase is the data pipeline."
**If asked, "what is the 'none' class for?":** "So the system can say *'that wasn't
one of our signs'* instead of being forced to pick an emergency phrase by mistake."

### Slide 6 — Literature Review (overview table)
**Say:** "We surveyed the field, and a clear pattern emerges. Almost all existing
systems — including the Nepali ones — work only on *static* images or single alphabet
letters, and several have real weaknesses: data leakage, or needing special gloves.
There's a public NSL dataset from 2024, but it's alphabet-only, and nobody built a
recognition system on it. So the gap is obvious: no one has built a *signer-
independent, motion-based, emergency-phrase* recognizer for NSL. That's exactly our
space."
**Cue:** This is the most important lit slide — deliver the "gap" line clearly. Spend
your time here; the next five slides are supporting detail.

### Slide 7 — Ligal & Baral (2022)
**Say:** "Ligal and Baral explored deep models for NSL — a CNN with an RNN, and a CNN
with a Vision Transformer. It confirms there's research interest in NSL, but they use
the heavy pixel-based approach and don't target emergencies or real-time use."

### Slide 8 — Goyal & Velmathi (2023)
**Say:** "This one is the closest in *method* to ours. They applied MediaPipe Holistic
— pose plus both hands — to Indian Sign Language, with LSTMs for moving signs. It
validates our exact direction: landmarks plus a temporal model. The difference is
it's for Indian Sign Language, and it produces no audio output."

### Slide 9 — Sunuwar, Borah & Kharga (2024), NSL23
**Say:** "This paper released NSL23 — the first public NSL dataset, 630 videos of the
49 alphabet characters from 14 volunteers. It proves NSL data collection is feasible,
and that video matters because some signs involve motion. But it covers only the
alphabet, not phrases."

### Slide 10 — Shrestha et al. (2024)
**Say:** "This is the closest *prior NSL landmark* work. They detect NSL letters with
MediaPipe, but then turn the landmarks back into skeleton images for a CNN. Accuracy
is high — but the same signers appear in both training and testing, so it's not
signer-independent, and it's static, letters only."
**Cue:** Stress "not signer-independent" — it's the weakness your design fixes.

### Slide 11 — Poudel et al. (2025)
**Say:** "The most recent — a 36-class static NSL image dataset with MobileNetV2 and
ResNet50, around 90% accuracy. But it's single-image and pixel-based, and it excludes
exactly the motion-based phrase signs we care about."

### Slide 12 — Methodology: organizations contacted
**Say:** "For our methodology, we started with the community, not the code. We
identified and reached out to Deaf organizations and schools across the valley —
district associations, special-needs schools, and the national federation. We visited
two in particular, highlighted here: RCRD & CBR Bhaktapur, and the National Federation
of the Deaf Nepal."

### Slide 13 — RCRD & CBR Bhaktapur (field visit)
**Say:** "Here we are at RCRD & CBR Bhaktapur, a community-based rehabilitation
organization. We went there for two reasons: to collect data, and to *learn the
correct signs directly from Deaf experts*. Because their time was limited, the experts
demonstrated and contributed around ten clips for each sign — that gave us an
authentic, expert reference for every phrase. We then recorded the remaining clips
ourselves at college to complete the dataset."
**Cue:** Emphasize *learning the signs from the experts* — it's what makes your data
authentic rather than guessed.

### Slide 14 — National Federation of the Deaf Nepal (NFDN)
**Say:** "We did the same at the National Federation of the Deaf Nepal — the national
apex body for the Deaf community. Learning and recording the signs with the experts
there is what gives us confidence that our phrases are correct, real-world NSL, and not
something we invented. Between the two visits, the expert reference clips anchor the
whole dataset."

### Slide 15 — The recorder tool (at Khwopa College)
**Say:** "Back at college, we built our own recording tool — this is the NSL Landmark
Recorder we developed, and it's what we used to record the rest of the clips ourselves.
The signer picks an ID and a phrase, then records one sign at a time. You can see
MediaPipe tracking the body pose and both hands live — those colored dots and lines are
the landmark points. For every frame we save 225 numbers, and we store *only* those
points, never the video."
**Cue:** Point at the skeleton overlay on the hands/body in the screenshot.

### Slide 16 — Outcomes
**Say:** "The outcome of the collection stage: a complete dataset. Seven classes — the
six emergency phrases plus 'none' — and 570 usable clips in total, recorded across
four different signers, kept roughly balanced so no class or person dominates."

### Slide 17 — Methodology: the full pipeline
**Say:** "This diagram is our whole pipeline, in two stages. **Stage one, collection**
— the webcam captures the signer, MediaPipe detects the pose and both hands giving 225
values per frame, and we save each clip raw, with quality metadata. Clips are
naturally different lengths. Then in the middle, we **scan every clip's frame count and
pick one sequence length at the 95th percentile** — that came out to 148 frames.
**Stage two, preprocessing** — we first **normalize** every frame, then **standardize
the length**: long clips are evenly subsampled, short clips are padded. The output is
clean **training tensors** — data, mask, and labels. Model training is the next phase."
**Cue:** Trace the boxes left to right with your hand as you speak; pause at the
orange "sequence length" branch and the two green preprocessing boxes.
**If asked, "why the 95th percentile?":** "It keeps about 95% of clips essentially
complete, and only the rare very long clips get compressed — instead of forcing every
clip to pad up to the single longest one, which would be mostly empty."

### Slide 18 — Result
**Say:** "Here are the results of the pipeline. Every clip is now normalized and
standardized, and the training tensors are generated — the data is ready for the
model. On the left is *how* we normalize. For each **hand**, we re-center on the wrist
and scale by the wrist-to-knuckle distance — so it doesn't matter where the hand is or
how big it looks. For the **body**, we center on the midpoint of the shoulders and
scale by shoulder width. This strips out the signer's position and size, so the model
learns the *gesture itself*. On the right are our actual outputs: **X**, the input
tensor, is 570 clips × 148 frames × 225 features; and **y** is the 570 labels."
**If asked, "why normalize at all?":** "The same sign looks like completely different
numbers if the person stands closer, further, or off to one side. Normalization
removes all of that, leaving only the movement — so the model isn't distracted by
where someone was standing."
**If asked, "why two different anchors?":** "The shoulder anchor keeps *where the hand
is relative to the body* — chest versus side. The wrist anchor captures *finger shape*
on its own. One anchor couldn't do both."

### Slide 19 — Conclusion & Future Work
**Say:** "To conclude: two things are **done** — the data pipeline is complete and
tested, and the dataset is collected and processed. What's **next**: signer-independent
evaluation, to prove it works on people it has never seen; then live webcam
recognition; and finally browser-based deployment, so anyone can use it without
installing anything."
**Cue:** Point to the two green check-marks (done) versus the three "Next" items.

### Slide 20 — Thank You
**Say:** "That's where our project stands at the mid-term. Thank you — we'd be happy to
take your questions."
**Cue:** Smile, stop talking, and let the panel take over. Have
[MID_DEFENSE_PREPROCESSING.md](MID_DEFENSE_PREPROCESSING.md) §9 fresh in your mind.

---

## The five questions you're most likely to get

Short, confident answers (full versions in
[MID_DEFENSE_PREPROCESSING.md](MID_DEFENSE_PREPROCESSING.md) §9):

1. **"Why landmarks instead of the video/pixels?"** — 225 numbers per frame instead of
   ~900,000; it ignores background, lighting, and appearance; needs far less data; and
   runs on a CPU or in a browser.
2. **"Where's the model / the accuracy?"** — Model training is the next phase; this
   checkpoint is the data pipeline, which is complete and tested.
3. **"How is this different from the existing NSL work?"** — Prior NSL work is static,
   alphabet-only, and not signer-independent. Ours is motion-based emergency phrases,
   with a signer-labeled dataset built for signer-independent testing.
4. **"How did you choose 148 frames?"** — The 95th percentile of real clip lengths —
   keeps most clips complete while only compressing the rare long ones.
5. **"Who performed the signs / is the data reliable?"** — Deaf experts at RCRD & CBR
   Bhaktapur and NFDN taught us the signs and contributed about 10 reference clips per
   sign; we recorded the rest ourselves at college. The dataset is balanced across
   signers, with per-clip hand-detection quality metrics recorded for every clip.
