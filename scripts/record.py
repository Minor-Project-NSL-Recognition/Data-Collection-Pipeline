"""NSL landmark recorder — captures RAW, variable-length MediaPipe Holistic clips.

Does NOT normalize or standardize (that is build_dataset.py's job), so seq_len and
the normalization scheme stay cheap, re-runnable knobs. Saves per-clip landmarks
(frames, 225) + quality metadata under <output_dir>/raw/<class>/.

Frames are mirror-flipped before detection (natural for the signer); inference must
apply the same flip, and left/right hand blocks refer to the mirrored image.

    python scripts/record.py
"""

import json
import os
import sys
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

from nslr import config as C
from nslr.landmarks import extract_frame_vector

CAM_INDEX = 0
PREVIEW_WIDTH = 720

# Remembers last output folder + signer ID across runs (machine-local, gitignored).
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".recorder_config.json")


def _load_last_settings():
    try:
        with open(CONFIG_PATH, "r") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NSL Landmark Recorder")

        last = _load_last_settings()
        self.output_dir = tk.StringVar(value=last.get("output_dir", ""))
        self.signer_id = tk.StringVar(value=last.get("signer_id", "signer01"))
        self.current_class = tk.StringVar(value=list(C.CLASSES.keys())[0])
        self.status = tk.StringVar(value="Choose an output folder to begin.")

        self.recording = False
        self.buffer = []          # 225-vectors for the clip in progress
        self.frame_flags = []     # per-frame detection flags, for the quality report
        self.record_start = None

        self._build_ui()

        self.output_dir.trace_add("write", lambda *_: self._save_settings())
        self.signer_id.trace_add("write", lambda *_: self._save_settings())

        if self.output_dir.get() and os.path.isdir(self.output_dir.get()):
            self._ensure_class_folders(self.output_dir.get())
            self.status.set(f"Restored last output folder — {len(C.CLASSES)} class folders ready.")
            self._refresh_counts()
        elif self.output_dir.get():
            self.status.set("Last output folder no longer exists — choose a new one.")
            self.output_dir.set("")
        if self._init_camera():
            self._update_frame()
        self._auto_refresh_counts()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")

        self.video_label = ttk.Label(main)
        self.video_label.grid(row=0, column=0, rowspan=8, padx=(0, 12))

        ttk.Label(main, text="Output folder", font=("", 10, "bold")).grid(row=0, column=1, sticky="w")
        folder_row = ttk.Frame(main)
        folder_row.grid(row=1, column=1, sticky="ew", pady=(0, 10))
        ttk.Entry(folder_row, textvariable=self.output_dir, width=28).pack(side="left")
        ttk.Button(folder_row, text="Browse…", command=self._pick_folder).pack(side="left", padx=(4, 0))

        ttk.Label(main, text="Signer ID", font=("", 10, "bold")).grid(row=2, column=1, sticky="w")
        ttk.Entry(main, textvariable=self.signer_id, width=36).grid(row=3, column=1, sticky="w", pady=(0, 10))

        ttk.Label(main, text="Phrase (class)", font=("", 10, "bold")).grid(row=4, column=1, sticky="w")
        # Radios occupy rows 5 .. 5+n-1; everything below is offset by the class
        # count so adding/removing a class (e.g. the 'none' negatives) never collides.
        for i, (key, label) in enumerate(C.CLASSES.items()):
            ttk.Radiobutton(main, text=label, value=key, variable=self.current_class,
                            command=self._release_focus).grid(row=5 + i, column=1, sticky="w")
        base = 5 + len(C.CLASSES)

        controls = ttk.Frame(main)
        controls.grid(row=base, column=1, sticky="w", pady=(12, 0))
        self.btn_start = ttk.Button(controls, text="● Start recording", command=self.start_recording)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(controls, text="■ Stop & save", command=self.stop_recording, state="disabled")
        self.btn_stop.pack(side="left", padx=(6, 0))
        self.btn_discard = ttk.Button(controls, text="Discard", command=self.discard_recording, state="disabled")
        self.btn_discard.pack(side="left", padx=(6, 0))

        ttk.Separator(main, orient="horizontal").grid(row=base + 1, column=1, sticky="ew", pady=8)
        ttk.Label(main, textvariable=self.status, wraplength=320).grid(row=base + 2, column=1, sticky="w")

        self.counts = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.counts, wraplength=320, foreground="#555").grid(
            row=base + 3, column=1, sticky="w", pady=(6, 0))

        self.root.bind("<space>", self._on_space)
        self.root.bind("<Escape>", self._on_escape)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_space(self, event):
        if event.widget.winfo_class() in ("TEntry", "Entry"):
            return
        self._toggle()
        return "break"

    def _on_escape(self, event):
        if event.widget.winfo_class() in ("TEntry", "Entry"):
            return
        self.discard_recording()
        return "break"

    def _toggle(self):
        self.stop_recording() if self.recording else self.start_recording()

    def _release_focus(self):
        # Keep spacebar toggling single-purpose: return focus to root after any
        # button/radiobutton click so <space> doesn't re-invoke that widget too.
        self.root.focus_set()

    def _save_settings(self):
        try:
            with open(CONFIG_PATH, "w") as fh:
                json.dump({"output_dir": self.output_dir.get(), "signer_id": self.signer_id.get()}, fh)
        except OSError:
            pass

    def _ensure_class_folders(self, path):
        os.makedirs(os.path.join(path, "raw"), exist_ok=True)
        for key in C.CLASSES:
            os.makedirs(os.path.join(path, "raw", key), exist_ok=True)

    def _pick_folder(self):
        path = filedialog.askdirectory(title="Choose the project data folder")
        if path:
            self.output_dir.set(path)
            self._ensure_class_folders(path)
            self.status.set(f"Output set. raw/ created with {len(C.CLASSES)} class folders.")
            self._refresh_counts()
        self._release_focus()

    def _refresh_counts(self):
        base = self.output_dir.get()
        if not base:
            self.counts.set("")
            return
        parts = []
        for key in C.CLASSES:
            d = os.path.join(base, "raw", key)
            n = len([f for f in os.listdir(d) if f.endswith(".npy")]) if os.path.isdir(d) else 0
            parts.append(f"{key}: {n}")
        self.counts.set("Clips on disk — " + " | ".join(parts))

    def _auto_refresh_counts(self):
        # Poll disk so counts stay in sync with files removed outside the app.
        self._refresh_counts()
        self.root.after(1500, self._auto_refresh_counts)

    def _init_camera(self):
        self.cap = cv2.VideoCapture(CAM_INDEX)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", f"Could not open camera {CAM_INDEX}.")
            self.root.destroy()
            return False
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
        self.drawing = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles
        return True

    def _update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(15, self._update_frame)
            return

        frame = cv2.flip(frame, 1)                       # mirror, natural for the signer
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)
        rgb.flags.writeable = True

        if self.recording:
            vec, flags = extract_frame_vector(results)
            self.buffer.append(vec)
            self.frame_flags.append(flags)

        self._show(self._draw(frame, results))
        self.root.after(10, self._update_frame)

    def _draw(self, frame, results):
        mp_h = mp.solutions.holistic
        self.drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_h.POSE_CONNECTIONS,
            landmark_drawing_spec=self.styles.get_default_pose_landmarks_style())
        for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
            self.drawing.draw_landmarks(
                frame, hand, mp_h.HAND_CONNECTIONS,
                landmark_drawing_spec=self.styles.get_default_hand_landmarks_style())

        if self.recording:
            n = len(self.buffer)
            elapsed = time.time() - self.record_start
            cv2.circle(frame, (24, 24), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"REC  {n} frames  {elapsed:0.1f}s", (44, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    def _show(self, frame):
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (PREVIEW_WIDTH, int(h * PREVIEW_WIDTH / w)))
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.video_label.configure(image=photo)
        self.video_label.image = photo   # keep a reference or it gets garbage-collected

    def start_recording(self):
        if self.recording:
            return
        if not self.output_dir.get():
            messagebox.showwarning("No output folder", "Choose an output folder first.")
            self._release_focus()
            return
        if not self.signer_id.get().strip():
            messagebox.showwarning("No signer ID", "Enter a signer ID first.")
            self._release_focus()
            return

        self.buffer = []
        self.frame_flags = []
        self.record_start = time.time()
        self.recording = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_discard.config(state="normal")
        self.status.set(f"Recording  →  {self.current_class.get()}  /  {self.signer_id.get()}")
        self._release_focus()

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_discard.config(state="disabled")

        if len(self.buffer) < 5:
            self.status.set(f"Clip too short ({len(self.buffer)} frames) — discarded.")
            self.buffer, self.frame_flags = [], []
            self._release_focus()
            return

        self._save_clip()
        self._release_focus()

    def discard_recording(self):
        if not self.recording:
            return
        self.recording = False
        n = len(self.buffer)
        self.buffer, self.frame_flags = [], []
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_discard.config(state="disabled")
        self.status.set(f"Discarded ({n} frames).")
        self._release_focus()

    def _save_clip(self):
        cls = self.current_class.get()
        signer = self.signer_id.get().strip().replace(" ", "_")
        folder = os.path.join(self.output_dir.get(), "raw", cls)
        os.makedirs(folder, exist_ok=True)

        # Next index = max existing + 1 (not count + 1, which would overwrite after a deletion).
        prefix = f"{cls}__{signer}__"
        indices = []
        for f in os.listdir(folder):
            if f.startswith(prefix) and f.endswith(".npy"):
                try:
                    indices.append(int(f[len(prefix):-len(".npy")]))
                except ValueError:
                    pass
        idx = max(indices, default=0) + 1
        stem = f"{prefix}{idx:03d}"

        clip = np.stack(self.buffer).astype(np.float32)   # (frames, 225)
        np.save(os.path.join(folder, stem + ".npy"), clip)

        n = len(self.frame_flags)
        meta = {
            "class": cls, "signer": signer, "clip_index": idx, "n_frames": n,
            "duration_sec": round(time.time() - self.record_start, 2),
            "feature_dim": C.FEATURE_DIM,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "pose_detect_rate": round(sum(f["pose"] for f in self.frame_flags) / n, 3),
            "left_hand_detect_rate": round(sum(f["left_hand"] for f in self.frame_flags) / n, 3),
            "right_hand_detect_rate": round(sum(f["right_hand"] for f in self.frame_flags) / n, 3),
            "any_hand_detect_rate": round(
                sum(f["left_hand"] or f["right_hand"] for f in self.frame_flags) / n, 3),
        }
        with open(os.path.join(folder, stem + ".json"), "w") as fh:
            json.dump(meta, fh, indent=2)

        warn = "  ⚠ LOW HAND DETECTION — consider re-recording." if meta["any_hand_detect_rate"] < 0.7 else ""
        self.status.set(f"Saved {stem}.npy  —  {n} frames, hands {meta['any_hand_detect_rate']:.0%}{warn}")
        self.buffer, self.frame_flags = [], []
        self._refresh_counts()

    def _on_close(self):
        try:
            self.cap.release()
            self.holistic.close()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    RecorderApp(root)
    root.mainloop()
