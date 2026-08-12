"""Plays the pre-recorded Nepali phrase for a recognised sign.

Deliberately dependency-free. The venv this runs in is a pinned, fragile stack
(tensorflow 2.17 / numpy 1.26 / jax 0.4.31 / mediapipe 0.10.14) where adding a
package to play a 2-second WAV is a poor trade, so on Windows this uses
`winsound` from the standard library and elsewhere shells out to whatever the
platform already ships.

That choice is why scripts/build_audio.py emits 16-bit PCM WAV: `winsound` plays
that and nothing else -- not the Ogg/Opus the phone recorded, not MP3.

Playback is always asynchronous. scripts/live_demo.py calls this from inside its
capture loop, which also drives `cv2.imshow`; a blocking play would freeze the
video for the length of the phrase, right at the moment an audience is watching.

Nepali TEXT is not handled here, and the OpenCV demo does not display any: PIL
in this venv is built without raqm, so it cannot shape Devanagari (pre-base
matras land on the wrong side of their consonant and conjuncts fall apart). The
Flutter app has HarfBuzz and shows the script properly.
"""

import os
import subprocess
import sys
import threading

from . import config as C

AUDIO_DIR = os.path.join(C.REPO_ROOT, "app", "assets", "audio", "ne")

# Spoken for a rejected clip (open-set gate or the `none` class). Not a class
# key -- see scripts/build_audio.py's REJECT_KEY.
REJECT_KEY = "unknown"


def _posix_player():
    """First available CLI audio player, or None."""
    from shutil import which
    for cmd in (["afplay"], ["paplay"], ["aplay", "-q"], ["ffplay", "-nodisp",
                                                          "-autoexit", "-loglevel", "quiet"]):
        if which(cmd[0]):
            return cmd
    return None


class Voice:
    """Class key -> spoken Nepali phrase.

    Missing clips are not an error. Recording six takes is a manual job that
    proceeds one phrase at a time, and a half-finished set should still let the
    demo run -- the phrases that have audio speak, the rest stay silent.
    """

    def __init__(self, class_names, audio_dir=AUDIO_DIR, verbose=True):
        self.dir = audio_dir
        self.clips = {}
        missing = []
        for key in class_names:
            # `none` is the open-set negative class. Speaking anything for it
            # would announce a phrase at a signer who did not sign one, so it is
            # excluded here as well as having no file on disk.
            if key == "none":
                continue
            path = os.path.join(audio_dir, f"{key}.wav")
            (self.clips.setdefault(key, path) if os.path.exists(path)
             else missing.append(key))
        self.missing = missing

        # Rejection feedback lives alongside the phrase clips but is loaded
        # separately: it is not in class_names and must not gate `self.clips`.
        reject_path = os.path.join(audio_dir, f"{REJECT_KEY}.wav")
        self.reject_clip = reject_path if os.path.exists(reject_path) else None

        self._posix = None if sys.platform == "win32" else _posix_player()
        have_any = bool(self.clips) or self.reject_clip is not None
        self.enabled = have_any and (sys.platform == "win32"
                                     or self._posix is not None)

        if verbose:
            if not self.clips:
                print(f"No Nepali audio in {audio_dir} — running silent. "
                      "Build it with: python scripts/build_audio.py")
            elif missing:
                print(f"Nepali audio: {len(self.clips)} of "
                      f"{len(self.clips) + len(missing)} phrases. "
                      f"Silent: {', '.join(missing)}")
            else:
                print(f"Nepali audio: all {len(self.clips)} phrases loaded.")
            print(f"Rejection clip: {'loaded' if self.reject_clip else 'missing'} "
                  f"({REJECT_KEY}.wav)")
            if have_any and not self.enabled:
                print("  ...but no audio player found (tried afplay/paplay/"
                      "aplay/ffplay) — playback disabled.")

    def play(self, key):
        """Speak `key`'s phrase. Returns True if a clip was actually started.

        Never raises: a demo must not die because an audio device is busy.
        """
        return self._play_path(self.clips.get(key), key)

    def play_reject(self):
        """Speak the "not recognised" clip for a rejected clip (open-set gate
        or the `none` class). Same never-raises contract as `play`."""
        return self._play_path(self.reject_clip, REJECT_KEY)

    def _play_path(self, path, label):
        if path is None or not self.enabled:
            return False
        try:
            if sys.platform == "win32":
                import winsound
                # SND_ASYNC returns immediately and supersedes whatever is
                # already playing, which is the behaviour we want when signs
                # come faster than the clips finish. SND_NODEFAULT stops Windows
                # from substituting its own beep if the file cannot be read.
                winsound.PlaySound(
                    path,
                    winsound.SND_FILENAME | winsound.SND_ASYNC
                    | winsound.SND_NODEFAULT,
                )
            else:
                threading.Thread(
                    target=lambda: subprocess.run(
                        self._posix + [path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    ),
                    daemon=True,
                ).start()
            return True
        except Exception as exc:              # noqa: BLE001 - never fatal
            print(f"audio playback failed for {label}: {exc}")
            return False

    def stop(self):
        """Cut playback short, e.g. when the window is closing."""
        if sys.platform == "win32" and self.enabled:
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:                 # noqa: BLE001
                pass
