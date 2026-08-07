"""One streaming capture session: JPEG frames in, a landmark clip out.

Two things here are correctness-critical rather than incidental.

**Decimation happens BEFORE MediaPipe, not after.** The model reads frame count
as duration, and the training clips were captured at ~15.7 fps (see
config.TRAIN_CAPTURE_FPS). A phone streaming at 30 fps would hand the model twice
the frames for the same sign. Dropping the extra frames up front both fixes that
and halves the server's work -- and it means Holistic's internal tracker sees the
same ~16 fps cadence it saw during recording, which is the state it was tracking
at when the training landmarks were produced.

**Frames are mirrored**, matching record.py's `cv2.flip(frame, 1)`. Get this
wrong and the left/right hand blocks swap places in the 225-vector.
"""

import time

import cv2
import numpy as np

from nslr import config as C
from nslr.landmarks import extract_frame_vector

MAX_FRAMES = 600          # ~38 s at 15.7 fps; guards against an unbounded stream
MAX_JPEG_BYTES = 2 << 20  # 2 MB per frame is already absurd for 640x480


class StreamSession:
    def __init__(self, target_fps=None, mirror=True):
        import mediapipe as mp

        self.target_fps = float(target_fps or C.TRAIN_CAPTURE_FPS)
        self.period_ms = 1000.0 / self.target_fps
        self.mirror = mirror
        # model_complexity=1 is not a tuning knob: it is what record.py ran, and
        # complexity 0 is a different network producing different landmarks.
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

        self.frames = []
        self.received = 0
        self.dropped = 0
        self._next_due = None
        self._t0 = time.monotonic()
        self.closed = False

    def _now_ms(self):
        return (time.monotonic() - self._t0) * 1000.0

    def _due(self, t_ms):
        """Rate-limit to target_fps using a due-time accumulator.

        A naive `t - last >= period` test aliases badly: a client at 16 fps
        against a 15.7 fps target sits just under the interval every time and
        gets halved to 8 fps. Accumulating the due time instead keeps the long-run
        rate correct regardless of the client's cadence.
        """
        if self._next_due is None:
            self._next_due = t_ms + self.period_ms
            return True
        if t_ms < self._next_due:
            return False
        self._next_due += self.period_ms
        if self._next_due <= t_ms:          # stream fell behind — resync
            self._next_due = t_ms + self.period_ms
        return True

    def add_jpeg(self, blob, timestamp_ms=None):
        """Decode + (maybe) extract one frame. Returns a small status dict."""
        if self.closed:
            raise RuntimeError("session is closed")
        self.received += 1

        if len(blob) > MAX_JPEG_BYTES:
            self.dropped += 1
            return {"accepted": False, "reason": "frame_too_large", "n_frames": len(self.frames)}

        t_ms = float(timestamp_ms) if timestamp_ms is not None else self._now_ms()
        if not self._due(t_ms):
            self.dropped += 1
            return {"accepted": False, "reason": "rate_limited", "n_frames": len(self.frames)}
        if len(self.frames) >= MAX_FRAMES:
            self.dropped += 1
            return {"accepted": False, "reason": "max_frames", "n_frames": len(self.frames)}

        bgr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            self.dropped += 1
            return {"accepted": False, "reason": "decode_failed", "n_frames": len(self.frames)}

        if self.mirror:
            bgr = cv2.flip(bgr, 1)                       # same mirror as record.py
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        vector, flags = extract_frame_vector(self.holistic.process(rgb))
        self.frames.append(vector)

        return {"accepted": True, "n_frames": len(self.frames),
                "pose": flags["pose"],
                "hands": bool(flags["left_hand"] or flags["right_hand"])}

    def clip(self):
        """(n_frames, 225) float32, in record.py's raw un-normalized form."""
        if not self.frames:
            return np.zeros((0, C.FEATURE_DIM), dtype=np.float32)
        return np.stack(self.frames).astype(np.float32)

    def stats(self):
        elapsed = self._now_ms() / 1000.0
        return {"received": self.received, "kept": len(self.frames), "dropped": self.dropped,
                "elapsed_sec": round(elapsed, 2),
                "effective_fps": round(len(self.frames) / elapsed, 1) if elapsed > 0 else None,
                "target_fps": self.target_fps}

    def reset(self):
        self.frames = []
        self.received = 0
        self.dropped = 0
        self._next_due = None
        self._t0 = time.monotonic()

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.holistic.close()
            except Exception:
                pass
