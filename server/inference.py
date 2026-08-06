"""Model loading + the decision rule, shared by every endpoint.

Deliberately reuses nslr.preprocess rather than reimplementing it: the whole
point of serving from Python is that the request path is byte-for-byte the
pipeline the 99.7% number was measured on. Any divergence here silently
invalidates that number.

Runs the TFLite export by preference (no TensorFlow in the container, ~900 KB
instead of a 2.3 MB Keras file plus a 3 GB dependency), falling back to
model.keras when TFLite is unavailable.
"""

import json
import os

import numpy as np

from nslr import config as C
from nslr import ood
from nslr.preprocess import normalize_clip, standardize_length


def _load_interpreter(path):
    """tflite_runtime if present (small Linux wheel), else TensorFlow's copy."""
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_path=path)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=path)
    except ImportError as exc:
        raise RuntimeError(
            "Neither tflite_runtime nor tensorflow is installed — cannot run model.tflite"
        ) from exc


class Predictor:
    """Raw (n_frames, 225) landmark clip -> a decision."""

    def __init__(self, results_dir=C.RESULTS_DIR):
        with open(os.path.join(results_dir, "model_meta.json")) as fh:
            meta = json.load(fh)
        self.class_names = meta["class_names"]
        self.seq_len = int(meta["seq_len"])
        self.threshold = float(meta.get("confidence_threshold", 0.75))
        self.n_classes = len(self.class_names)

        tflite_path = os.path.join(results_dir, "model.tflite")
        keras_path = os.path.join(results_dir, "model.keras")
        if os.path.exists(tflite_path):
            self.backend = "tflite"
            self._interp = _load_interpreter(tflite_path)
            self._interp.allocate_tensors()
            self._in = self._interp.get_input_details()[0]
            self._out = self._interp.get_output_details()
        elif os.path.exists(keras_path):
            self.backend = "keras"
            from tensorflow import keras
            model = keras.models.load_model(keras_path)
            self._model = keras.Model(model.inputs,
                                      [model.layers[-1].output, model.layers[-2].output])
        else:
            raise SystemExit(
                f"No model in {results_dir}. Run scripts/train_model.py, then "
                f"scripts/export_tflite.py")

        # Open-set gate. Optional: without it the 7-way softmax is closed-world
        # and will confidently name a phrase for any input at all.
        self.ood = None
        ood_json = os.path.join(results_dir, "ood.json")
        ood_npz = os.path.join(results_dir, "ood_stats.npz")
        if os.path.exists(ood_json):
            with open(ood_json) as fh:
                d = json.load(fh)
            self.ood = (np.array(d["means"]), np.array(d["precision"]), float(d["threshold"]))
        elif os.path.exists(ood_npz):
            means, precision, thr, _ = ood.load_stats(ood_npz)
            self.ood = (means, precision, thr)

        # The stored threshold is the p99 of *training* distances, so it assumes
        # inference looks like recording did. A phone, an unseen signer and a
        # lower capture fps all push embeddings outward, and a real sign can
        # then land past a threshold fitted without any of them. This override
        # lets that be corrected against the deployment in front of you without
        # a retrain. `off` disables the gate entirely — a closed-world softmax
        # that names a phrase for any input at all, including nonsense.
        self.ood_threshold_source = "fitted"
        override = os.environ.get("NSL_OOD_THRESHOLD", "").strip()
        if override and self.ood is not None:
            if override.lower() in ("off", "none", "0"):
                self.ood = None
                self.ood_threshold_source = "disabled"
            else:
                try:
                    means, precision, _ = self.ood
                    self.ood = (means, precision, float(override))
                    self.ood_threshold_source = "override"
                except ValueError:
                    pass

    def _forward(self, fixed):
        """(seq_len, 225) -> (probs (n_classes,), embedding (d,))."""
        x = fixed[None, ...].astype(np.float32)
        if self.backend == "tflite":
            self._interp.set_tensor(self._in["index"], x)
            self._interp.invoke()
            got = [self._interp.get_tensor(o["index"]) for o in self._out]
            # by shape, not order — TFLite output naming is a converter artifact
            probs = next(g for g in got if g.shape[-1] == self.n_classes)[0]
            emb = next(g for g in got if g.shape[-1] != self.n_classes)[0]
            return probs, emb
        probs, emb = self._model.predict(x, verbose=0)
        return probs[0], emb[0]

    def predict(self, clip):
        """clip: (n_frames, 225) RAW landmarks, exactly as record.py saved them."""
        clip = np.asarray(clip, dtype=np.float32)
        if clip.ndim != 2 or clip.shape[1] != C.FEATURE_DIM:
            raise ValueError(f"expected (n_frames, {C.FEATURE_DIM}), got {clip.shape}")
        if len(clip) < 5:
            raise ValueError(f"clip too short: {len(clip)} frames")

        fixed, _, mode = standardize_length(normalize_clip(clip), self.seq_len)
        probs, emb = self._forward(fixed)

        order = np.argsort(probs)[::-1]
        label = self.class_names[int(order[0])]
        confidence = float(probs[order[0]])

        distance, is_unknown = None, False
        if self.ood is not None:
            means, precision, thr = self.ood
            distance = float(ood.mahalanobis_min(emb[None, :], means, precision)[0][0])
            is_unknown = distance > thr

        if is_unknown:
            status = "unknown"
        elif confidence >= self.threshold:
            status = "accepted"
        else:
            status = "low_confidence"

        return {
            "status": status,
            "label": label if status == "accepted" else None,
            "display": C.CLASSES.get(label, label) if status == "accepted" else None,
            "confidence": confidence,
            "best_guess": label,
            "top3": [{"label": self.class_names[i], "confidence": float(probs[i])}
                     for i in order[:3]],
            "ood_distance": distance,
            "n_frames": int(len(clip)),
            "standardize": mode,
        }
