"""Export model.keras -> model.tflite (+ portable OOD stats), and verify the export.

Why the conversion looks convoluted: on this repo's pinned TF 2.17 + Keras 3 stack,
both `TFLiteConverter.from_keras_model()` and `from_saved_model()` abort the whole
process inside MLIR --

    error: missing attribute 'value'
    LLVM ERROR: Failed to infer result type(s)

-- which is a hard crash, not a Python exception you can catch. Freezing the
variables into constants first (`convert_variables_to_constants_v2`) sidesteps it.
The result needs only TFLITE_BUILTINS, so no Flex delegate and no bloat.

Exports TWO outputs -- softmax AND the penultimate Dense-32 embedding -- because
the open-set rejection gate (nslr/ood.py) scores distance in that embedding space.
Without it the server could classify but never say "unknown sign".

    python scripts/export_tflite.py
    python scripts/export_tflite.py --skip-verify        # if X.npy isn't built

Outputs to <results>/:
    model.tflite    ~900 KB, builtin ops only
    ood.json        prototypes + precision + threshold, framework-independent
"""

import argparse
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nslr import config as C
from nslr import ood


def build_predictor(model):
    """Two-output view of the trained model: [softmax, penultimate embedding]."""
    from tensorflow import keras
    return keras.Model(model.inputs, [model.layers[-1].output, model.layers[-2].output])


def convert(predictor, seq_len, n_features):
    """Freeze variables, then convert. Returns the flatbuffer bytes."""
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    fn = tf.function(lambda x: predictor(x, training=False))
    concrete = fn.get_concrete_function(tf.TensorSpec([1, seq_len, n_features], tf.float32))
    frozen = convert_variables_to_constants_v2(concrete)

    conv = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return conv.convert()


def verify(tflite_path, predictor, processed_dir, n_classes, n=30):
    """Compare TFLite against Keras on real clips. Raises if they disagree."""
    import tensorflow as tf

    x_path = os.path.join(processed_dir, "X.npy")
    if not os.path.exists(x_path):
        print(f"  ! {x_path} missing — skipping verification")
        return None

    X = np.load(x_path)
    idx = np.linspace(0, len(X) - 1, min(n, len(X))).astype(int)

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()

    worst, agree = 0.0, []
    for i in idx:
        x = X[i][None].astype(np.float32)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        got = [interp.get_tensor(o["index"]) for o in outs]
        # identify by shape, not by name/order — those are converter artifacts
        probs = [g for g in got if g.shape[-1] == n_classes][0][0]
        ref = predictor.predict(x, verbose=0)[0][0]
        worst = max(worst, float(np.abs(probs - ref).max()))
        agree.append(bool(probs.argmax() == ref.argmax()))

    ok = np.mean(agree)
    print(f"  verified on {len(idx)} clips: worst |dp| = {worst:.2e}, argmax agreement {ok:.0%}")
    if worst > 1e-4 or ok < 1.0:
        raise SystemExit(f"Export does NOT match Keras (|dp|={worst:.2e}, agree={ok:.0%}). "
                         "Do not ship this file.")
    return {"worst_abs_prob_diff": worst, "argmax_agreement": float(ok), "n_clips": len(idx)}


def main():
    p = argparse.ArgumentParser(description="Export the BiLSTM to TFLite for serving.")
    p.add_argument("--results", default=C.RESULTS_DIR)
    p.add_argument("--processed", default=C.PROCESSED_DIR)
    p.add_argument("--skip-verify", action="store_true")
    a = p.parse_args()

    model_path = os.path.join(a.results, "model.keras")
    meta_path = os.path.join(a.results, "model_meta.json")
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        raise SystemExit(f"No model in {a.results}. Run:  python scripts/train_model.py")

    from tensorflow import keras
    model = keras.models.load_model(model_path)
    with open(meta_path) as fh:
        meta = json.load(fh)

    seq_len, n_features = model.input_shape[1], model.input_shape[2]
    n_classes = model.output_shape[-1]
    print(f"model: seq_len={seq_len} features={n_features} classes={n_classes}")

    predictor = build_predictor(model)
    blob = convert(predictor, seq_len, n_features)
    out_path = os.path.join(a.results, "model.tflite")
    with open(out_path, "wb") as fh:
        fh.write(blob)
    print(f"  wrote {out_path}  ({len(blob)/1024:.0f} KB)")

    check = None
    if not a.skip_verify:
        check = verify(out_path, predictor, a.processed, n_classes)

    # --- OOD stats as JSON so the server (and a future Dart port) need no numpy format ---
    ood_npz = os.path.join(a.results, "ood_stats.npz")
    ood_json_path = os.path.join(a.results, "ood.json")
    if os.path.exists(ood_npz):
        means, precision, threshold, percentile = ood.load_stats(ood_npz)
        with open(ood_json_path, "w") as fh:
            json.dump({"means": means.tolist(), "precision": precision.tolist(),
                       "threshold": threshold, "percentile": percentile,
                       "embedding_dim": int(means.shape[1]),
                       "class_names": meta["class_names"]}, fh)
        print(f"  wrote {ood_json_path}  ({os.path.getsize(ood_json_path)/1024:.0f} KB, "
              f"reject distance > {threshold:.2f})")
    else:
        print(f"  ! {ood_npz} missing — server will run without open-set rejection")

    meta["tflite"] = "model.tflite"
    meta["tflite_verified"] = check
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  updated {meta_path}")


if __name__ == "__main__":
    main()
