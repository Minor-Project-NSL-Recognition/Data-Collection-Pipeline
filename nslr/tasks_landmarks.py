"""MediaPipe Tasks (PoseLandmarker + HandLandmarker) -> the same 225-vector.

`landmarks.py` builds the 225-vector from Holistic, which is what the model was
trained on. Holistic is a legacy solution with no mobile/Flutter equivalent, so
the phone app has to assemble the same vector from two independent Tasks
detectors instead. This module is that assembly, kept in `nslr/` so the parity
experiment and the eventual Dart port check against one definition.

The contract is identical to `landmarks.extract_frame_vector`:
    [pose 99 | left hand 63 | right hand 63], missing blocks zero-filled.

The one genuinely ambiguous piece is which detected hand goes in which block.
Holistic never has to decide — it crops each hand from the *pose's* wrist, so
"left_hand_landmarks" means "the hand on the pose's left wrist" by construction.
Tasks detects hands standalone and labels them via a handedness classifier. We
reproduce Holistic's rule (nearest pose wrist) and carry the classifier's answer
alongside, so `parity_report.py` can measure how often the two disagree.
"""

import numpy as np

from . import config as C

POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16


class TasksExtractor:
    """Stateful per-frame extractor. Feed RGB frames with increasing timestamps.

    Uses RunningMode.VIDEO so both detectors track across frames, matching how
    Holistic behaves inside record.py's capture loop.
    """

    def __init__(self, pose_model_path, hand_model_path,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 static_image_mode=False):
        """static_image_mode=True runs each frame independently (RunningMode.IMAGE).

        Use it for offline analysis of unrelated stills: in VIDEO mode both
        detectors carry tracking state forward, so a previous unrelated image
        silently biases the next one.
        """
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._vision = vision
        self.static = static_image_mode
        mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO

        pose_opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=pose_model_path),
            running_mode=mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=hand_model_path),
            running_mode=mode,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.pose = vision.PoseLandmarker.create_from_options(pose_opts)
        self.hands = vision.HandLandmarker.create_from_options(hand_opts)

    def extract(self, rgb, timestamp_ms):
        """RGB uint8 frame -> (225-vector, flags). Same shape/order as Holistic's."""
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.static:
            pose_res = self.pose.detect(image)
            hand_res = self.hands.detect(image)
        else:
            pose_res = self.pose.detect_for_video(image, timestamp_ms)
            hand_res = self.hands.detect_for_video(image, timestamp_ms)

        pose = np.zeros(C.POSE_DIM, dtype=np.float32)
        left = np.zeros(C.HAND_DIM, dtype=np.float32)
        right = np.zeros(C.HAND_DIM, dtype=np.float32)
        flags = {"pose": False, "left_hand": False, "right_hand": False,
                 "label_agrees": None}

        pose_pts = None
        if pose_res.pose_landmarks:
            lms = pose_res.pose_landmarks[0]
            pose_pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            pose = pose_pts.flatten()
            flags["pose"] = True

        if hand_res.hand_landmarks:
            blocks, agrees = self._assign_hands(hand_res, pose_pts)
            flags["label_agrees"] = agrees
            if blocks["left"] is not None:
                left = blocks["left"]
                flags["left_hand"] = True
            if blocks["right"] is not None:
                right = blocks["right"]
                flags["right_hand"] = True

        vector = np.concatenate([pose, left, right])
        assert vector.shape[0] == C.FEATURE_DIM, f"expected {C.FEATURE_DIM}, got {vector.shape[0]}"
        return vector, flags

    def _assign_hands(self, hand_res, pose_pts):
        """Route each detected hand into the left/right block.

        Primary rule = nearest pose wrist (reproduces Holistic). Returns the
        blocks plus whether the handedness classifier picked the same side, so
        disagreement is measurable rather than assumed away.
        """
        hands = []
        for i, lms in enumerate(hand_res.hand_landmarks):
            pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            label = ""
            if hand_res.handedness and i < len(hand_res.handedness):
                label = hand_res.handedness[i][0].category_name  # "Left" / "Right"
            hands.append((pts, label))

        blocks = {"left": None, "right": None}
        agrees = None

        if pose_pts is not None:
            # Nearest-wrist assignment in image xy (z scales differ between the
            # pose and hand models, so distance uses xy only).
            anchors = {"left": pose_pts[POSE_LEFT_WRIST, :2],
                       "right": pose_pts[POSE_RIGHT_WRIST, :2]}
            claims = []
            for pts, label in hands:
                wrist = pts[C.HAND_WRIST, :2]
                d_left = float(np.linalg.norm(wrist - anchors["left"]))
                d_right = float(np.linalg.norm(wrist - anchors["right"]))
                claims.append(("left" if d_left <= d_right else "right",
                               min(d_left, d_right), pts, label))
            # Both hands claiming one wrist: the closer one wins, other flips.
            claims.sort(key=lambda c: c[1])
            taken = set()
            for side, _, pts, label in claims:
                slot = side if side not in taken else ("right" if side == "left" else "left")
                if slot in taken:
                    continue
                taken.add(slot)
                blocks[slot] = pts.flatten()
                if label:
                    agrees = (label.lower() == slot) if agrees is None else (
                        agrees and label.lower() == slot)
        else:
            # No pose this frame -> fall back to the handedness classifier.
            for pts, label in hands:
                slot = label.lower() if label.lower() in ("left", "right") else None
                if slot and blocks[slot] is None:
                    blocks[slot] = pts.flatten()

        return blocks, agrees

    def close(self):
        for det in (self.pose, self.hands):
            try:
                det.close()
            except Exception:
                pass
