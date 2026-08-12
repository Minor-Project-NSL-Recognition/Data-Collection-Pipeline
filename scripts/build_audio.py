"""Turn recorded Nepali voice clips into the WAVs the app and demo play.

Source recordings live in `assets/audio/src/<class_key>.<ext>` and are whatever
the phone's voice recorder produced -- in practice Opus-in-Ogg, because that is
what WhatsApp and Telegram voice notes are. Those cannot ship as-is:

  * iOS AVFoundation supports neither the Ogg container nor the Opus codec, so
    an .ogg asset is silent on iPhone while working fine on Android.
  * Python's `winsound`, which scripts/live_demo.py uses precisely because it is
    stdlib and cannot disturb the pinned venv, plays PCM WAV and nothing else.

So this script decodes to 16-bit PCM mono WAV at 48 kHz -- the one format every
target plays without a decoder dependency, at the rate phone recordings already
use, so nothing is resampled on the way in.

It also does the two things that are tedious by hand and obvious when skipped:

  * **Trims silence** from both ends. Leading silence is pure added latency
    between the sign finishing and the phrase being heard.
  * **Normalizes loudness** to -16 LUFS (the usual target for speech on mobile)
    with a two-pass EBU R128 measurement. The first four recordings spanned 11
    LU -- `call_police` at -28.4 against `building_on_fire` at -17.0 -- which in
    an emergency app means one phrase is inaudible in the situations that
    matter. Peak normalization would not have fixed it; the quiet clip also had
    a low peak.

Class keys come from app/assets/phrases.json, the same file config.py and the
Flutter app read, so this script cannot drift from the phrase list either. A key
with no recording is reported and skipped -- the app treats a missing clip as
"that phrase stays silent" rather than failing, so partial sets are fine while
recording is still in progress.

REJECT_KEY ("unknown") is built the same way but is NOT a phrase key: it is not
in phrases.json, has no class, and is never chosen by the model. It is what
scripts/live_demo.py speaks when a clip is rejected (open-set gate or the
`none` class) -- audible "that didn't match" feedback, not a claimed answer, so
the ban on `none` having audio does not apply to it.

    python scripts/build_audio.py            # convert everything present
    python scripts/build_audio.py --check    # report status, write nothing

Requires ffmpeg on PATH.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from nslr import config as C  # noqa: E402

SRC_DIR = os.path.join(C.REPO_ROOT, "assets", "audio", "src")
OUT_DIR = os.path.join(C.REPO_ROOT, "app", "assets", "audio", "ne")

# Anything ffmpeg can decode; listed in preference order when a key has several.
SRC_EXTS = (".ogg", ".opus", ".m4a", ".mp3", ".wav", ".aac", ".flac")

# Spoken once for any rejected clip (open-set reject or the `none` class). Not a
# phrase key -- see the module docstring.
REJECT_KEY = "unknown"

# Speech on mobile. -1.5 dBTP leaves headroom so the lossy path to the speaker
# cannot clip on the loudest syllable.
TARGET_I, TARGET_TP, TARGET_LRA = -16.0, -1.5, 11.0

# Must be set explicitly. `loudnorm` runs its analysis at 192 kHz internally,
# and without an -ar the encoder inherits that rate instead of the source's --
# which silently produced 192 kHz assets four times larger than needed, and
# inconsistently, since loudnorm only resamples on some of its internal paths.
OUT_RATE = 48000

# Trim near-silence off the head (keeping 50 ms so the first consonant is not
# clipped) and off the tail (keeping 150 ms, which sounds less abrupt than a
# hard cut). The areverse sandwich is how ffmpeg trims a tail: silenceremove
# only ever works on the start of a stream.
TRIM = (
    "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-45dB,"
    "areverse,"
    "silenceremove=start_periods=1:start_silence=0.15:start_threshold=-45dB,"
    "areverse"
)

# Evens out the level *within* a take before the clip-wide gain is applied.
#
# Without it the phone recordings could not reach the -16 LUFS target at all:
# every one of them hit the -1.5 dBTP ceiling first and stalled 2-4 dB short,
# because a spoken phrase recorded on a handset has a ~18 dB crest factor -- a
# couple of plosive spikes sitting far above the words around them. The peak
# ceiling then rations the gain for the whole clip on behalf of two consonants.
# Raising the ceiling instead would only buy ~1 dB and risk clipping.
#
# speechnorm is ffmpeg's filter for this specific problem, and it is applied
# BEFORE loudnorm so the measurement pass sees the levelled signal.
SPEECHNORM = "speechnorm=e=12.5:r=0.0001:l=1"


def run(args):
    """ffmpeg/ffprobe, returning combined output. Raises on a non-zero exit."""
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"{args[0]} failed:\n{p.stderr[-2000:]}")
    return (p.stdout or "") + (p.stderr or "")


def _loudnorm_json(path, prefilters):
    out = run([
        "ffmpeg", "-hide_banner", "-nostdin", "-i", path,
        "-af", prefilters + f"loudnorm=I={TARGET_I}:TP={TARGET_TP}"
                            f":LRA={TARGET_LRA}:print_format=json",
        "-f", "null", "-",
    ])
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", out, re.S)
    if not match:
        raise RuntimeError(f"could not parse loudnorm output for {path}")
    return json.loads(match.group(0))


def measure(path):
    """Pass 1: EBU R128 stats for `path`, measured through the same trim and
    levelling the output gets -- measuring the untrimmed signal would let
    leading silence drag the integrated loudness down and push the clip too
    loud, and measuring before speechnorm would describe a signal that no
    longer exists by the time the gain is applied."""
    return _loudnorm_json(path, f"{TRIM},{SPEECHNORM},")


def measure_written(path):
    """Loudness of a finished file, with NO pre-filters.

    Deliberately not `measure`: that one reports what the input looks like after
    the processing chain, so pointing it at an already-processed file describes
    a hypothetical second pass rather than the bytes on disk."""
    return float(_loudnorm_json(path, "")["input_i"])


def convert(src, dst):
    """Pass 2: trim, normalize using the measured values, write PCM WAV."""
    m = measure(src)
    # linear=true applies one constant gain instead of compressing dynamically,
    # which is what keeps a short spoken phrase sounding like the original take.
    loudnorm = (
        f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}"
        f":measured_I={m['input_i']}:measured_TP={m['input_tp']}"
        f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
        f":offset={m['target_offset']}:linear=true:print_format=summary"
    )
    run([
        "ffmpeg", "-hide_banner", "-nostdin", "-y", "-i", src,
        "-af", f"{TRIM},{SPEECHNORM},{loudnorm}",
        "-ar", str(OUT_RATE), "-ac", "1", "-c:a", "pcm_s16le", dst,
    ])
    return float(m["input_i"])


def probe_duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", path])
    return float(out.strip())


def find_source(key):
    for ext in SRC_EXTS:
        path = os.path.join(SRC_DIR, key + ext)
        if os.path.exists(path):
            return path
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="report which phrases have recordings; write nothing")
    a = ap.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SystemExit("ffmpeg and ffprobe must be on PATH.")

    # `none` is the open-set negative class. It must never get a clip: playing
    # it as a claimed answer is the failure this whole gate exists to prevent.
    # REJECT_KEY is added separately -- it speaks "rejected", not a phrase.
    keys = [k for k in C.PHRASES if k != "none"] + [REJECT_KEY]
    os.makedirs(OUT_DIR, exist_ok=True)

    stray = os.path.join(SRC_DIR, "none.ogg")
    if os.path.exists(stray):
        raise SystemExit(f"Refusing to build: {stray} exists. `none` is the "
                         "rejection class and must stay silent.")

    have, missing = [], []
    for key in keys:
        src = find_source(key)
        (have if src else missing).append((key, src))

    if a.check:
        print(f"{len(have)}/{len(keys)} phrases have a recording in {SRC_DIR}")
        for key, src in have:
            print(f"  ok      {key:<18} {os.path.basename(src)}")
        for key, _ in missing:
            print(f"  MISSING {key}")
        return

    print(f"Building {len(have)} clip(s) into app/assets/audio/ne/")
    for key, src in have:
        dst = os.path.join(OUT_DIR, key + ".wav")
        before = convert(src, dst)
        after = measure_written(dst)
        size_kb = os.path.getsize(dst) / 1024
        print(f"  {key:<18} {before:>7.1f} -> {after:>6.1f} LUFS   "
              f"{probe_duration(dst):.2f}s   {size_kb:.0f} KB")

    if missing:
        print(f"\nStill to record: {', '.join(k for k, _ in missing)}")
        print("Drop <key>.ogg (or .m4a/.wav) into assets/audio/src/ and re-run.")
        print("Until then those phrases display text but play no audio.")


if __name__ == "__main__":
    main()
