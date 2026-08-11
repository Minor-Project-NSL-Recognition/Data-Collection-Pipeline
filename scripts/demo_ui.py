"""Presentation chrome for scripts/live_demo.py.

The demo used to stamp a translucent text box onto the camera image. This module
composes the window instead — header, camera panel, results sidebar, footer —
so the only things drawn *over* the video are the landmark overlay and the
recording badge. Everything else has a place to live.

Two notes on the implementation:

* **Text** is drawn with Pillow and a real UI font when both are available, and
  falls back to OpenCV's Hershey font when they are not. The layout is identical
  either way, so a machine without Pillow still gets the same demo, just with a
  more dated typeface. Draw calls are queued and rendered in one pass, because
  the BGR<->RGB round trip is the expensive part and one per frame is fine.
* **Static chrome** (background, rules, headings, footer) is rendered once into
  `self.base`; each frame is a memcpy of that plus the dynamic bits. Drawing the
  whole layout every frame would cost more than the model does.

Strings stay ASCII on purpose: the Hershey fallback cannot render `·`, `≥` or
`…` and silently substitutes boxes.
"""

import math
import os

import cv2
import numpy as np

# --------------------------------------------------------------- palette ---
# BGR, because that is what OpenCV wants. Light neutral surface, one accent,
# three status colours. The status colours are the darker end of their ramps so
# they still pass as text on white — the bright versions wash out on a page.
BG = (248, 246, 245)  # window background   #F5F6F8
SURFACE = (255, 255, 255)  # cards
SURFACE_2 = (236, 231, 228)  # bar tracks          #E4E7EC
LINE = (227, 221, 217)  # hairlines, borders  #D9DDE3
TEXT = (29, 24, 21)  # #15181D
MUTED = (109, 98, 91)  # #5B626D
DIM = (156, 145, 138)  # #8A919C
ACCENT = (235, 111, 31)  # #1F6FEB
OK = (75, 127, 26)  # #1A7F4B
WARN = (9, 83, 180)  # #B45309
BAD = (47, 47, 209)  # #D12F2F

# config.TRAIN_CAPTURE_FPS is 15.7; below this the loop is dropping frames the
# model expects to see, so the readout stops being decorative.
SLOW_FPS = 12.0

# Header pill + card accent per result state.
STATES = {
    "ready": ("READY", DIM),
    "rec": ("RECORDING", BAD),
    "ok": ("MATCH", OK),
    "low": ("LOW CONFIDENCE", WARN),
    "ood": ("REJECTED", BAD),
    "short": ("TOO SHORT", WARN),
}


def lerp(a, b, t):
    """Blend two BGR tuples."""
    t = max(0.0, min(1.0, t))
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def split_label(key, label):
    """'1. I can't breathe (Medical)' -> ('I can't breathe', 'Medical').

    config.CLASSES stores GUI labels with an index prefix and a bracketed
    category. The sidebar wants those as separate fields, not one long string.
    """
    if key == "none":
        return "No known sign", "negative"
    text = (label or key).strip()
    head, _, rest = text.partition(".")
    if head.strip().isdigit():
        text = rest.strip()
    category = ""
    if text.endswith(")") and "(" in text:
        text, _, category = text.rpartition("(")
        category = category[:-1].strip()
    return text.strip() or key, category


# ------------------------------------------------------------------ text ---
_FONT_DIRS = (
    r"C:\Windows\Fonts",
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts/TTF",
    "/Library/Fonts",
    "/System/Library/Fonts",
)
_FONT_NAMES = {
    "regular": (
        "segoeui.ttf",
        "Inter-Regular.ttf",
        "Roboto-Regular.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
        "arial.ttf",
    ),
    "bold": (
        "segoeuisb.ttf",
        "segoeuib.ttf",
        "Inter-SemiBold.ttf",
        "Roboto-Medium.ttf",
        "DejaVuSans-Bold.ttf",
        "LiberationSans-Bold.ttf",
        "arialbd.ttf",
    ),
    "mono": (
        "consola.ttf",
        "JetBrainsMono-Regular.ttf",
        "DejaVuSansMono.ttf",
        "LiberationMono-Regular.ttf",
        "cour.ttf",
    ),
}
_MPL_FALLBACK = {
    "regular": "DejaVuSans.ttf",
    "bold": "DejaVuSans-Bold.ttf",
    "mono": "DejaVuSansMono.ttf",
}


def _find_font(role):
    for name in _FONT_NAMES[role]:
        for directory in _FONT_DIRS:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
    try:  # matplotlib is already a dependency and ships DejaVu
        import matplotlib

        path = os.path.join(
            matplotlib.get_data_path(), "fonts", "ttf", _MPL_FALLBACK[role]
        )
        if os.path.exists(path):
            return path
    except Exception:
        pass
    return None


def _hershey(size, role):
    """Map a pixel size to (fontScale, thickness) for the OpenCV fallback."""
    return size / 30.0, max(1, int(round(size / (12.0 if role == "bold" else 17.0))))


class Type:
    """Queued text renderer. `draw()` records, `flush()` paints them all at once.

    y is always the top of the line, so callers can lay out by adding heights
    without caring which backend is active.
    """

    def __init__(self):
        self._pil = None
        self._paths = {}
        self._fonts = {}
        self._queue = []
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            return  # no Pillow -> Hershey, layout unchanged
        paths = {role: _find_font(role) for role in _FONT_NAMES}
        if paths.get("regular"):
            self._pil = (Image, ImageDraw, ImageFont)
            self._paths = paths

    @property
    def truetype(self):
        return self._pil is not None

    def _font(self, size, role):
        font = self._fonts.get((size, role))
        if font is None:
            path = self._paths.get(role) or self._paths["regular"]
            font = self._fonts[(size, role)] = self._pil[2].truetype(path, size)
        return font

    def measure(self, text, size, role="regular"):
        """Advance width in pixels."""
        if not text:
            return 0
        if self._pil:
            font = self._font(size, role)
            try:
                return int(round(font.getlength(text)))
            except AttributeError:  # Pillow < 8
                return font.getsize(text)[0]
        scale, thick = _hershey(size, role)
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]

    def ellipsize(self, text, size, role, max_w):
        if self.measure(text, size, role) <= max_w:
            return text
        while text and self.measure(text + "...", size, role) > max_w:
            text = text[:-1]
        return text.rstrip() + "..."

    def wrap(self, text, size, role, max_w, max_lines=2):
        lines, current = [], ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if not current or self.measure(candidate, size, role) <= max_w:
                current = candidate
                continue
            lines.append(current)
            current = word
            if len(lines) == max_lines - 1:
                break
        lines.append(current)
        if len(lines) == max_lines:
            lines[-1] = self.ellipsize(lines[-1], size, role, max_w)
        return lines[:max_lines]

    def draw(self, text, xy, size, color, role="regular", align="left"):
        if text:
            self._queue.append(
                (str(text), int(xy[0]), int(xy[1]), int(size), color, role, align)
            )

    def flush(self, img, box=None):
        """Paint and clear the queue. `box` limits the Pillow round trip to the
        region the queued text lives in — converting the whole canvas (video
        included) four times a frame costs more than the model does, and the
        capture rate is not something this demo can afford to spend."""
        queue, self._queue = self._queue, []
        if not queue:
            return
        if self._pil:
            Image, ImageDraw, _ = self._pil
            if box is None:
                ox, oy, roi = 0, 0, img
            else:
                bx, by, bw, bh = (int(v) for v in box)
                ox, oy = max(0, bx), max(0, by)
                roi = img[
                    oy : min(img.shape[0], by + bh), ox : min(img.shape[1], bx + bw)
                ]
            pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            for text, x, y, size, color, role, align in queue:
                if align != "left":
                    width = self.measure(text, size, role)
                    x -= width if align == "right" else width // 2
                draw.text(
                    (x - ox, y - oy),
                    text,
                    font=self._font(size, role),
                    fill=(color[2], color[1], color[0]),
                    anchor="la",
                )
            roi[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
            return
        for text, x, y, size, color, role, align in queue:
            scale, thick = _hershey(size, role)
            (width, height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick
            )
            if align != "left":
                x -= width if align == "right" else width // 2
            cv2.putText(
                img,
                text,
                (x, y + height),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thick,
                cv2.LINE_AA,
            )


# ---------------------------------------------------------------- shapes ---
def rounded_rect(img, box, radius, color, thickness=-1):
    x, y, w, h = (int(v) for v in box)
    r = max(0, min(int(radius), w // 2, h // 2))
    if thickness < 0:
        if r:
            cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
            cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
            for cx, cy in (
                (x + r, y + r),
                (x + w - r, y + r),
                (x + r, y + h - r),
                (x + w - r, y + h - r),
            ):
                cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        return
    cv2.line(img, (x + r, y), (x + w - r, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y + r), (x, y + h - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness, cv2.LINE_AA)
    for (cx, cy), angle in (
        ((x + r, y + r), 180),
        ((x + w - r, y + r), 270),
        ((x + w - r, y + h - r), 0),
        ((x + r, y + h - r), 90),
    ):
        cv2.ellipse(img, (cx, cy), (r, r), angle, 0, 90, color, thickness, cv2.LINE_AA)


def bar(img, box, frac, color, track=SURFACE_2):
    """Rounded progress track with a rounded fill."""
    x, y, w, h = (int(v) for v in box)
    rounded_rect(img, (x, y, w, h), h // 2, track)
    filled = int(round(max(0.0, min(1.0, frac)) * w))
    if filled >= 2:
        rounded_rect(img, (x, y, filled, h), h // 2, color)


# ------------------------------------------------------------- dashboard ---
class Dashboard:
    """Composes one demo frame. Call `render()` per frame, `imshow` the result."""

    def __init__(self, frame_w, frame_h, *, labels, seq_len, threshold, cam_index):
        self.frame_w, self.frame_h = frame_w, frame_h
        self.labels = labels  # class key -> (phrase, category)
        self.seq_len = seq_len
        self.threshold = threshold
        self.type = Type()

        # One design unit == 1px at 640x480. Scaling off the *height* keeps the
        # sidebar's stacked sections fitting beside 16:9 capture too.
        self.u = max(0.7, min(3.0, frame_h / 480.0))
        px = self.px
        self.pad, self.gap = px(16), px(14)
        self.head_h, self.foot_h = px(48), px(34)
        self.side_w = px(258)

        self.vx, self.vy = self.pad, self.head_h + self.pad
        self.sx, self.sy = self.vx + frame_w + self.gap, self.vy
        self.sh = frame_h
        self.w = self.sx + self.side_w + self.pad
        self.h = self.head_h + self.pad + frame_h + self.pad + self.foot_h

        # Sidebar section tops, resolved once so base/render agree.
        y = self.sy
        self.y_pred_label, y = y, y + px(18)
        self.y_pred_card, y = y, y + px(116) + px(20)
        self.y_top_label, y = y, y + px(18)
        self.row_h = px(30)
        self.y_rows, y = y, y + 3 * self.row_h
        self.y_meta = self.sy + self.sh - px(2 * 17)
        self.show_meta = self.y_meta > y + px(24)

        # Only the header and the sidebar are repainted from `base` each frame:
        # the video area is overwritten by the blit, and the footer never
        # changes. Restoring the whole canvas is a pointless 4 MB memcpy.
        self.head_box = (0, 0, self.w, self.head_h + 1)
        self.side_box = (
            self.sx - self.gap // 2,
            self.head_h,
            self.w - self.sx + self.gap // 2 + self.pad,
            self.h - self.foot_h - self.head_h,
        )

        self.base = np.empty((self.h, self.w, 3), np.uint8)
        self._build_base(cam_index)
        self.canvas = self.base.copy()

    # -- helpers -------------------------------------------------------------
    def px(self, v):
        return max(1, int(round(v * self.u)))

    def window_size(self, cap=1600):
        """Default window size: the canvas, shrunk to fit a projector."""
        if self.w <= cap:
            return self.w, self.h
        return cap, int(round(self.h * cap / self.w))

    # -- static chrome -------------------------------------------------------
    def _build_base(self, cam_index):
        px, t = self.px, self.type
        self.base[:] = BG
        cv2.line(self.base, (0, self.head_h), (self.w, self.head_h), LINE, 1)
        cv2.line(
            self.base,
            (0, self.h - self.foot_h),
            (self.w, self.h - self.foot_h),
            LINE,
            1,
        )

        # Header identity.
        cy = self.head_h // 2
        x = self.pad + px(16)
        title = "NSL RECOGNITION"
        t.draw(title, (x, cy - px(8)), px(15), TEXT, "bold")

        # Sidebar section headings.
        for y, label in (
            (self.y_pred_label, "PREDICTION"),
            (self.y_top_label, "TOP MATCHES"),
        ):
            t.draw(label, (self.sx, y), px(10), DIM, "bold")

        # Static half of the spec block at the bottom of the sidebar.
        if self.show_meta:
            rows = [
                ("sequence", f"{self.seq_len} frames"),
                ("confidence", f">= {self.threshold:.0%}"),
            ]
            for i, (key, value) in enumerate(rows):
                y = self.y_meta + i * px(17)
                t.draw(key, (self.sx, y), px(11), DIM)
                t.draw(
                    value, (self.sx + self.side_w, y), px(11), MUTED, "mono", "right"
                )

        # Footer.
        fy = self.h - self.foot_h + (self.foot_h - px(11)) // 2
        t.draw(
            "SPACE  record / classify     R  reset     L  landmarks"
            "     F  fullscreen     Q  quit",
            (self.pad, fy),
            px(11),
            DIM,
        )
        t.draw(
            f"camera {cam_index}   {self.frame_w}x{self.frame_h}",
            (self.w - self.pad, fy),
            px(11),
            DIM,
            "mono",
            "right",
        )
        t.flush(self.base)

    # -- per-frame -----------------------------------------------------------
    def render(
        self,
        frame,
        *,
        recording=False,
        rec_frames=0,
        rec_time=0.0,
        result=None,
        fps=0.0,
        now=0.0,
    ):
        t = self.type
        for x, y, w, h in (self.head_box, self.side_box):
            self.canvas[y : y + h, x : x + w] = self.base[y : y + h, x : x + w]
        if frame.shape[:2] != (self.frame_h, self.frame_w):
            frame = cv2.resize(frame, (self.frame_w, self.frame_h))
        self.canvas[
            self.vy : self.vy + self.frame_h, self.vx : self.vx + self.frame_w
        ] = frame
        # Darker than LINE: this edge has to separate a bright camera image from
        # a near-white page, which a hairline tuned for the sidebar cannot do.
        cv2.rectangle(
            self.canvas,
            (self.vx - 1, self.vy - 1),
            (self.vx + self.frame_w, self.vy + self.frame_h),
            (205, 197, 191),
            1,
        )

        state = "rec" if recording else (result["state"] if result else "ready")
        self._header(state, fps)
        t.flush(self.canvas, self.head_box)
        self._prediction(state, result, rec_frames, rec_time, now)
        self._top_matches(state, result)
        t.flush(self.canvas, self.side_box)
        if recording:
            t.flush(self.canvas, self._rec_badge(rec_frames, rec_time, now))
        return self.canvas

    def _header(self, state, fps):
        px, t = self.px, self.type
        cy = self.head_h // 2
        fps_text = f"{fps:4.1f} fps"
        # Under-length clips get zero-padded, not resampled, so a slow loop is an
        # accuracy problem and not just a cosmetic one. Say so on screen.
        t.draw(
            fps_text,
            (self.w - self.pad, cy - px(6)),
            px(11),
            WARN if fps < SLOW_FPS else DIM,
            "mono",
            "right",
        )

        label, color = STATES[state]
        size = px(11)
        pw = t.measure(label, size, "bold") + px(30)
        ph = px(22)
        x = self.w - self.pad - t.measure(fps_text, px(11), "mono") - px(14) - pw
        rounded_rect(
            self.canvas, (x, cy - ph // 2, pw, ph), ph // 2, lerp(SURFACE, color, 0.22)
        )
        cv2.circle(self.canvas, (x + px(11), cy), px(3), color, -1, cv2.LINE_AA)
        t.draw(label, (x + px(20), cy - size // 2 - px(1)), size, color, "bold")

    def _prediction(self, state, result, rec_frames, rec_time, now):
        """The card that carries the answer. Idle and recording states occupy the
        same box so the layout never jumps mid-demo."""
        px, t = self.px, self.type
        x, y, w, h = self.sx, self.y_pred_card, self.side_w, px(116)
        color = STATES[state][1]

        rounded_rect(self.canvas, (x, y, w, h), px(8), SURFACE)
        # Fresh results glow for a moment, then settle back to a plain border.
        fresh = 0.0 if not result else max(0.0, 1.0 - (now - result["t"]) / 0.6)
        edge = (
            LINE
            if state in ("ready", "rec")
            else lerp(LINE, color, 0.35 + 0.65 * fresh)
        )
        rounded_rect(self.canvas, (x, y, w, h), px(8), edge, 1)

        ix, iw = x + px(14), w - px(28)
        if state == "ready":
            t.draw("Press SPACE to start recording", (ix, y + px(48)), px(11), DIM)
            bar(self.canvas, (ix, y + h - px(24), iw, px(5)), 0.0, DIM)
            return
        if state == "rec":
            t.draw("Recording", (ix, y + px(20)), px(19), TEXT, "bold")
            t.draw(
                f"{rec_frames} frames   {rec_time:0.1f}s",
                (ix, y + px(48)),
                px(11),
                MUTED,
                "mono",
            )
            # Progress toward the length the model was trained on.
            bar(
                self.canvas,
                (ix, y + h - px(24), iw, px(5)),
                rec_frames / max(1, self.seq_len),
                ACCENT,
            )
            t.draw(
                f"target {self.seq_len}",
                (x + w - px(14), y + h - px(42)),
                px(10),
                DIM,
                "mono",
                "right",
            )
            return

        lines = t.wrap(result["phrase"], px(19), "bold", iw, 2)
        for i, line in enumerate(lines):
            t.draw(
                line,
                (ix, y + px(18) + i * px(24)),
                px(19),
                TEXT if state == "ok" else color,
                "bold",
            )
        note_y = y + px(18) + len(lines) * px(24) + px(2)
        t.draw(result["note"], (ix, note_y), px(11), ACCENT if state == "ok" else MUTED)

        conf = result.get("conf")
        if conf is None:
            bar(self.canvas, (ix, y + h - px(24), iw, px(5)), 0.0, DIM)
            return
        pct = f"{conf:.0%}"
        t.draw(pct, (x + w - px(14), y + h - px(46)), px(15), color, "bold", "right")
        # A rejected clip still has a softmax winner; calling that "confidence"
        # next to "Unknown sign" reads as the UI contradicting itself.
        t.draw(
            "best guess" if state == "ood" else "confidence",
            (ix, y + h - px(43)),
            px(10),
            DIM,
        )
        bar(self.canvas, (ix, y + h - px(24), iw, px(5)), conf, color)

    def _top_matches(self, state, result):
        px, t = self.px, self.type
        top3 = (result or {}).get("top3") or []
        # After a rejection these are what softmax *wanted*, and the gate
        # overruled it — so they are shown, but never as the loudest thing here.
        overruled = state == "ood"
        for i in range(3):
            y = self.y_rows + i * self.row_h
            if i < len(top3):
                key, prob = top3[i]
                phrase = self.labels.get(key, (key, ""))[0]
                lead = MUTED if overruled else TEXT
                t.draw(
                    t.ellipsize(phrase, px(12), "regular", self.side_w - px(44)),
                    (self.sx, y),
                    px(12),
                    lead if i == 0 else MUTED,
                )
                t.draw(
                    f"{prob:.0%}",
                    (self.sx + self.side_w, y),
                    px(11),
                    MUTED if i else lead,
                    "mono",
                    "right",
                )
                fill = lerp(SURFACE_2, ACCENT, 0.45 if i or overruled else 1.0)
                bar(self.canvas, (self.sx, y + px(18), self.side_w, px(4)), prob, fill)
            else:
                t.draw("-", (self.sx, y), px(12), DIM)
                bar(self.canvas, (self.sx, y + px(18), self.side_w, px(4)), 0.0, DIM)

    def _rec_badge(self, frames, seconds, now):
        """The one overlay left on the video: it has to be where the signer looks."""
        px, t = self.px, self.type
        text = f"REC   {seconds:0.1f}s   {frames} frames"
        size = px(12)
        w = t.measure(text, size, "mono") + px(38)
        h = px(28)
        x, y = self.vx + px(12), self.vy + px(12)
        roi = self.canvas[y : y + h, x : x + w]
        patch = roi.copy()
        rounded_rect(patch, (0, 0, w - 1, h - 1), h // 2, (252, 251, 250))
        cv2.addWeighted(patch, 0.86, roi, 0.14, 0, roi)
        # A full-opacity hairline, so the badge still has an edge over a bright
        # feed — a pale translucent pill alone disappears into a lit background.
        rounded_rect(roi, (0, 0, w - 1, h - 1), h // 2, (198, 190, 185), 1)
        # Gentle pulse — enough to read as live, not enough to distract.
        pulse = 0.55 + 0.45 * math.sin(now * 5.0)
        cv2.circle(
            self.canvas,
            (x + px(15), y + h // 2),
            px(4),
            lerp((190, 190, 235), BAD, pulse),
            -1,
            cv2.LINE_AA,
        )
        t.draw(text, (x + px(25), y + (h - size) // 2 - px(1)), size, TEXT, "mono")
        return x, y, w, h
