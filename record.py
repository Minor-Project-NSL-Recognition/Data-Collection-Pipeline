"""
NSL Emergency-Phrase Recognition — Landmark Recording Tool
==========================================================

Records RAW, variable-length MediaPipe Holistic landmark sequences.

What this script deliberately does NOT do:
  - It does NOT normalize.
  - It does NOT standardize to seq_len.
Both belong in the separate processing step, so that seq_len and the
normalization scheme stay cheap, re-runnable knobs. This script only
captures the source-of-truth landmarks.

Output layout:
    <output_dir>/
        raw/
            cant_breathe/
                cant_breathe__signer01__001.npy      shape (frames, 225)
                cant_breathe__signer01__001.json     per-clip quality metadata
                ...
            building_on_fire/
            call_police/
            need_ambulance/
            help_danger/
            need_toilet/

Feature vector per frame (225 values, fixed order):
    [ pose 33 x (x,y,z) = 99 | left hand 21 x (x,y,z) = 63 | right hand 21 x (x,y,z) = 63 ]
Missing hand / pose blocks are zero-filled.

NOTE: frames are mirror-flipped (cv2.flip) BEFORE detection, on every frame.
This is consistent across the dataset, but it means inference must apply the
same flip, and "left"/"right" hand blocks refer to the mirrored image.

Requirements:
    pip install mediapipe opencv-python numpy pillow
    (Tkinter ships with most Python installs; on Debian/Ubuntu: sudo apt install python3-tk)
"""

import json
import os
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# The six emergency phrases. Keys are folder-safe class names; values are the
# labels shown in the GUI. Update these once the NSL consultant finalizes the
# gloss for each phrase.
CLASSES = {
    "cant_breathe":       "1. I can't breathe (Medical)",
    "building_on_fire":   "2. The building is on fire (Fire)",
    "call_police":        "3. Call the police (Crime)",
    "need_ambulance":     "4. I need an ambulance (Medical)",
    "help_danger":        "5. Help me / I am in danger (Generic)",
    "need_toilet":        "6. I need to go to the toilet (Basic need)",
}

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
FEATURE_DIM = POSE_LANDMARKS * 3 + HAND_LANDMARKS * 3 * 2   # 99 + 63 + 63 = 225

CAM_INDEX = 0
PREVIEW_WIDTH = 720


# --------------------------------------------------------------------------
# Landmark extraction
# --------------------------------------------------------------------------

def extract_frame_vector(results):
    """Convert one MediaPipe Holistic result into a 225-length float32 vector.

    Fixed order: pose (99) -> left hand (63) -> right hand (63).
    Any missing block is zero-filled, so every frame is always exactly 225 long.
    Returns (vector, flags) where flags records what was actually detected.
    """
    pose = np.zeros(POSE_LANDMARKS * 3, dtype=np.float32)
    left = np.zeros(HAND_LANDMARKS * 3, dtype=np.float32)
    right = np.zeros(HAND_LANDMARKS * 3, dtype=np.float32)

    flags = {"pose": False, "left_hand": False, "right_hand": False}

    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        flags["pose"] = True

    if results.left_hand_landmarks:
        left = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        flags["left_hand"] = True

    if results.right_hand_landmarks:
        right = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        flags["right_hand"] = True

    vector = np.concatenate([pose, left, right])
    assert vector.shape[0] == FEATURE_DIM, f"Expected {FEATURE_DIM}, got {vector.shape[0]}"
    return vector, flags


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------

class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NSL Landmark Recorder")

        self.output_dir = tk.StringVar(value="")
        self.signer_id = tk.StringVar(value="signer01")
        self.current_class = tk.StringVar(value=list(CLASSES.keys())[0])
        self.status = tk.StringVar(value="Choose an output folder to begin.")

        self.recording = False
        self.buffer = []          # list of 225-vectors for the clip in progress
        self.frame_flags = []     # detection flags per frame, for the quality report
        self.record_start = None

        self._build_ui()
        if self._init_camera():
            self._update_frame()
        self._auto_refresh_counts()

    # ---------------------------------------------------------------- UI

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")

        # Video preview
        self.video_label = ttk.Label(main)
        self.video_label.grid(row=0, column=0, rowspan=8, padx=(0, 12))

        # --- Output folder
        ttk.Label(main, text="Output folder", font=("", 10, "bold")).grid(
            row=0, column=1, sticky="w"
        )
        folder_row = ttk.Frame(main)
        folder_row.grid(row=1, column=1, sticky="ew", pady=(0, 10))
        ttk.Entry(folder_row, textvariable=self.output_dir, width=28).pack(side="left")
        ttk.Button(folder_row, text="Browse…", command=self._pick_folder).pack(
            side="left", padx=(4, 0)
        )

        # --- Signer
        ttk.Label(main, text="Signer ID", font=("", 10, "bold")).grid(
            row=2, column=1, sticky="w"
        )
        ttk.Entry(main, textvariable=self.signer_id, width=36).grid(
            row=3, column=1, sticky="w", pady=(0, 10)
        )

        # --- Class
        ttk.Label(main, text="Phrase (class)", font=("", 10, "bold")).grid(
            row=4, column=1, sticky="w"
        )
        for i, (key, label) in enumerate(CLASSES.items()):
            ttk.Radiobutton(
                main, text=label, value=key, variable=self.current_class
            ).grid(row=5 + i, column=1, sticky="w")

        # --- Record controls
        controls = ttk.Frame(main)
        controls.grid(row=11, column=1, sticky="w", pady=(12, 0))
        self.btn_start = ttk.Button(
            controls, text="● Start recording", command=self.start_recording
        )
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(
            controls, text="■ Stop & save", command=self.stop_recording, state="disabled"
        )
        self.btn_stop.pack(side="left", padx=(6, 0))
        self.btn_discard = ttk.Button(
            controls, text="Discard", command=self.discard_recording, state="disabled"
        )
        self.btn_discard.pack(side="left", padx=(6, 0))

        # --- Status / counts
        ttk.Separator(main, orient="horizontal").grid(
            row=12, column=1, sticky="ew", pady=8
        )
        ttk.Label(main, textvariable=self.status, wraplength=320).grid(
            row=13, column=1, sticky="w"
        )

        self.counts = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.counts, wraplength=320,
                  foreground="#555").grid(row=14, column=1, sticky="w", pady=(6, 0))

        self.root.bind("<space>", self._on_space)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_space(self, event):
        """Spacebar toggles recording — faster than clicking when signing."""
        # Ignore the shortcut while typing in a text field (signer ID / folder),
        # otherwise a space in the entry would also start a recording.
        if event.widget.winfo_class() in ("TEntry", "Entry"):
            return
        self._toggle()

    def _toggle(self):
        if self.recording:  
            
            self.stop_recording()
        else:
            self.start_recording()

    def _pick_folder(self):
        path = filedialog.askdirectory(title="Choose the project data folder")
        if path:
            self.output_dir.set(path)
            os.makedirs(os.path.join(path, "raw"), exist_ok=True)
            for key in CLASSES:
                os.makedirs(os.path.join(path, "raw", key), exist_ok=True)
            self.status.set(f"Output set. raw/ created with {len(CLASSES)} class folders.")
            self._refresh_counts()

    def _refresh_counts(self):
        """Show how many clips exist per class, so you can track collection progress."""
        base = self.output_dir.get()
        if not base:
            self.counts.set("")
            return
        parts = []
        for key in CLASSES:
            d = os.path.join(base, "raw", key)
            n = len([f for f in os.listdir(d) if f.endswith(".npy")]) if os.path.isdir(d) else 0
            parts.append(f"{key}: {n}")
        self.counts.set("Clips on disk — " + " | ".join(parts))

    def _auto_refresh_counts(self):
        """Poll the output folder periodically so the counts shown in the GUI
        stay in sync with whatever is actually on disk — including files or
        whole class folders removed outside the app (Explorer, another
        process, etc.), which the app would otherwise never learn about."""
        self._refresh_counts()
        self.root.after(1500, self._auto_refresh_counts)

    # ------------------------------------------------------------ Camera

    def _init_camera(self):
        """Open the camera and MediaPipe model. Returns False if the camera
        is unavailable, in which case the app shuts down."""
        self.cap = cv2.VideoCapture(CAM_INDEX)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", f"Could not open camera {CAM_INDEX}.")
            self.root.destroy()
            return False
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )
        self.drawing = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles
        return True

    def _update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(15, self._update_frame)
            return

        frame = cv2.flip(frame, 1)                       # mirror, feels natural to the signer
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)
        rgb.flags.writeable = True

        # Buffer landmarks only while recording
        if self.recording:
            vec, flags = extract_frame_vector(results)
            self.buffer.append(vec)
            self.frame_flags.append(flags)

        annotated = self._draw(frame, results)
        self._show(annotated)
        self.root.after(10, self._update_frame)

    def _draw(self, frame, results):
        mp_h = mp.solutions.holistic
        self.drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_h.POSE_CONNECTIONS,
            landmark_drawing_spec=self.styles.get_default_pose_landmarks_style(),
        )
        for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
            self.drawing.draw_landmarks(
                frame, hand, mp_h.HAND_CONNECTIONS,
                landmark_drawing_spec=self.styles.get_default_hand_landmarks_style(),
            )

        if self.recording:
            n = len(self.buffer)
            elapsed = time.time() - self.record_start
            cv2.circle(frame, (24, 24), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"REC  {n} frames  {elapsed:0.1f}s",
                        (44, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    def _show(self, frame):
        h, w = frame.shape[:2]
        scale = PREVIEW_WIDTH / w
        frame = cv2.resize(frame, (PREVIEW_WIDTH, int(h * scale)))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=photo)
        self.video_label.image = photo   # keep a reference or it gets garbage-collected

    # --------------------------------------------------------- Recording

    def start_recording(self):
        if self.recording:
            return
        if not self.output_dir.get():
            messagebox.showwarning("No output folder", "Choose an output folder first.")
            return
        if not self.signer_id.get().strip():
            messagebox.showwarning("No signer ID", "Enter a signer ID first.")
            return

        self.buffer = []
        self.frame_flags = []
        self.record_start = time.time()
        self.recording = True

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_discard.config(state="normal")
        self.status.set(f"Recording  →  {self.current_class.get()}  /  {self.signer_id.get()}")

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_discard.config(state="disabled")

        if len(self.buffer) < 5:
            self.status.set(f"Clip too short ({len(self.buffer)} frames) — discarded.")
            self.buffer = []
            self.frame_flags = []
            return

        self._save_clip()

    def discard_recording(self):
        self.recording = False
        n = len(self.buffer)
        self.buffer = []
        self.frame_flags = []
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_discard.config(state="disabled")
        self.status.set(f"Discarded ({n} frames).")

    def _save_clip(self):
        cls = self.current_class.get()
        signer = self.signer_id.get().strip().replace(" ", "_")
        folder = os.path.join(self.output_dir.get(), "raw", cls)
        os.makedirs(folder, exist_ok=True)

        # Next index for this class + signer. Use max existing index + 1, NOT
        # count + 1: after deleting a clip mid-sequence, count + 1 would
        # silently overwrite the last file.
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

        clip = np.stack(self.buffer).astype(np.float32)   # (frames, 225) — variable length
        np.save(os.path.join(folder, stem + ".npy"), clip)

        # Per-clip quality metadata. This is what lets you screen out bad clips
        # BEFORE they get stacked into X.npy.
        n = len(self.frame_flags)
        meta = {
            "class": cls,
            "signer": signer,
            "clip_index": idx,
            "n_frames": n,
            "duration_sec": round(time.time() - self.record_start, 2),
            "feature_dim": FEATURE_DIM,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            # Detection rates — a clip with a low hand rate is a candidate for rejection.
            "pose_detect_rate": round(sum(f["pose"] for f in self.frame_flags) / n, 3),
            "left_hand_detect_rate": round(sum(f["left_hand"] for f in self.frame_flags) / n, 3),
            "right_hand_detect_rate": round(sum(f["right_hand"] for f in self.frame_flags) / n, 3),
            "any_hand_detect_rate": round(
                sum(f["left_hand"] or f["right_hand"] for f in self.frame_flags) / n, 3
            ),
        }
        with open(os.path.join(folder, stem + ".json"), "w") as fh:
            json.dump(meta, fh, indent=2)

        warn = ""
        if meta["any_hand_detect_rate"] < 0.7:
            warn = "  ⚠ LOW HAND DETECTION — consider re-recording."

        self.status.set(
            f"Saved {stem}.npy  —  {n} frames, "
            f"hands {meta['any_hand_detect_rate']:.0%}{warn}"
        )
        self.buffer = []
        self.frame_flags = []
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
