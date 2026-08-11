import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'nslr_ood.dart';
import 'nslr_preprocess.dart';

/// Folder-safe class key -> spoken/displayed label. Mirrors `nslr/config.py`'s
/// CLASSES so the offline path speaks exactly what the server would have.
const Map<String, String> kClassLabels = {
  'cant_breathe': "1. I can't breathe (Medical)",
  'building_on_fire': '2. The building is on fire (Fire)',
  'call_police': '3. Call the police (Crime)',
  'need_ambulance': '4. I need an ambulance (Medical)',
  'help_danger': '5. Help me / I am in danger (Generic)',
  'need_toilet': '6. I need to go to the toilet (Basic need)',
  'none': '7. Unknown / none of the above (negatives)',
};

/// A guess with its probability, for the top-3 readout.
class Guess {
  Guess(this.label, this.confidence);
  final String label;
  final double confidence;
}

/// The same response shape the server returned, so the UI logic is unchanged.
class LocalResult {
  LocalResult({
    required this.status,
    required this.label,
    required this.display,
    required this.confidence,
    required this.bestGuess,
    required this.top3,
    required this.oodDistance,
    required this.nFrames,
    required this.framesUsed,
    required this.standardize,
    required this.inferenceMs,
  });

  /// `accepted` | `low_confidence` | `unknown`
  final String status;
  final String? label;
  final String? display;
  final double confidence;
  final String bestGuess;
  final List<Guess> top3;
  final double? oodDistance;

  /// Frames the camera actually produced.
  final int nFrames;

  /// Frames fed to the model after duration-based resampling. When this is much
  /// larger than [nFrames], the phone under-ran the target capture rate and the
  /// clip had to be stretched back to the training timescale.
  final int framesUsed;

  final String standardize;
  final int inferenceMs;

  bool get isAccepted => status == 'accepted';
}

/// Fully on-device recognition: raw 225-vectors in, a decision out.
///
/// This is the Dart port of `server/inference.py`. It runs the identical
/// sequence — normalize -> standardize -> BiLSTM -> Mahalanobis gate -> decision
/// rule — with no network involved.
class LocalRecognizer {
  LocalRecognizer._(
    this._interpreter,
    this._probsIndex,
    this._embIndex,
    this.classNames,
    this.seqLen,
    this.confidenceThreshold,
    this.ood,
  );

  final Interpreter _interpreter;
  final int _probsIndex;
  final int _embIndex;

  final List<String> classNames;
  final int seqLen;
  final double confidenceThreshold;

  /// Null when `ood.json` is absent, which leaves the softmax closed-world — it
  /// would then name a phrase for any input at all, including nonsense.
  final OodGate? ood;

  bool get hasOpenSetGate => ood != null;
  double? get oodThreshold => ood?.threshold;
  int get nClasses => classNames.length;

  static const _modelAsset = 'assets/models/model.tflite';
  static const _metaAsset = 'assets/models/model_meta.json';
  static const _oodAsset = 'assets/models/ood.json';

  /// Loads the model and gates from bundled assets.
  ///
  /// `seqLen` and the class list come from `model_meta.json`, never from a
  /// constant in this file. They have already changed twice during this project
  /// (151 -> 137 -> 146); hardcoding one would produce a shape mismatch at best
  /// and a silently mislabelled output at worst.
  static Future<LocalRecognizer> load() async {
    final metaRaw = await rootBundle.loadString(_metaAsset);
    final meta = jsonDecode(metaRaw) as Map<String, dynamic>;
    final classNames = (meta['class_names'] as List).cast<String>();
    final seqLen = (meta['seq_len'] as num).toInt();
    final confThreshold =
        ((meta['confidence_threshold'] as num?) ?? 0.75).toDouble();

    // fromBuffer, not fromAsset: asset-key resolution has differed across
    // tflite_flutter versions, and loading the bytes ourselves is unambiguous.
    final modelBytes = await rootBundle.load(_modelAsset);
    final interpreter = Interpreter.fromBuffer(
      modelBytes.buffer.asUint8List(
        modelBytes.offsetInBytes,
        modelBytes.lengthInBytes,
      ),
    );
    interpreter.allocateTensors();

    // Identify the two outputs by SHAPE, not by index or name. TFLite output
    // ordering is a converter artifact — `export_tflite.py` and
    // `server/inference.py` both make a point of this.
    final outputs = interpreter.getOutputTensors();
    var probsIndex = -1;
    var embIndex = -1;
    for (var i = 0; i < outputs.length; i++) {
      final last = outputs[i].shape.last;
      if (last == classNames.length) {
        probsIndex = i;
      } else {
        embIndex = i;
      }
    }
    if (probsIndex < 0) {
      throw StateError(
        'No output with ${classNames.length} classes; got '
        '${outputs.map((t) => t.shape).toList()}',
      );
    }

    final inputShape = interpreter.getInputTensor(0).shape;
    if (inputShape.length != 3 ||
        inputShape[1] != seqLen ||
        inputShape[2] != Nslr.featureDim) {
      throw StateError(
        'model input $inputShape does not match meta seq_len=$seqLen '
        'features=${Nslr.featureDim}',
      );
    }

    OodGate? gate;
    try {
      final oodRaw = await rootBundle.loadString(_oodAsset);
      gate = OodGate.fromJson(jsonDecode(oodRaw) as Map<String, dynamic>);
      // A gate fitted on a different model is worse than no gate: it rejects
      // real signs wholesale. This exact desync shipped in the repo once, because
      // ood.json is tracked in git while model.tflite is not, so a pull could
      // replace one without the other. Cheap to catch, expensive to debug.
      if (embIndex >= 0) {
        final embDim = interpreter.getOutputTensor(embIndex).shape.last;
        if (gate.embeddingDim != embDim) {
          throw StateError(
            'ood.json embedding dim ${gate.embeddingDim} != model $embDim — '
            'regenerate with scripts/export_tflite.py',
          );
        }
      }
      if (gate.nClasses != classNames.length) {
        throw StateError(
          'ood.json has ${gate.nClasses} prototypes for '
          '${classNames.length} classes',
        );
      }
    } catch (_) {
      // Absent or inconsistent: run closed-world rather than refusing to start,
      // and let the `gate` telemetry chip show it is off.
      gate = null;
    }

    return LocalRecognizer._(
      interpreter,
      probsIndex,
      embIndex,
      classNames,
      seqLen,
      confThreshold,
      gate,
    );
  }

  /// [clip] is raw, un-normalized 225-vectors exactly as the landmarker produced
  /// them — the same contract as `Predictor.predict`'s input.
  ///
  /// Mutates [clip] in place while normalizing; the caller must not reuse it.
  ///
  /// [durationSec] is the wall-clock length of the recording. Supplying it is
  /// strongly recommended: it lets the clip be resampled to the frame rate the
  /// model was trained at, which is what stops an under-performing phone from
  /// producing a clip the model reads as a much shorter, partial sign.
  ///
  /// [frameWidth]/[frameHeight] are the dimensions the landmarks were extracted
  /// from, used to undo the portrait/landscape aspect mismatch.
  /// [enforceOodGate] decides whether the Mahalanobis distance may REJECT a sign.
  /// It defaults to false, which is a deliberate deployment decision rather than
  /// laziness:
  ///
  /// The threshold is fitted as the p99 of *training* distances -- Holistic
  /// landmarks, a webcam, signers the model trained on. On-device the same real
  /// signs land at 20-27, past even the largest training distance (20.8), because
  /// the phone extracts landmarks with MediaPipe Tasks instead. Calibrating
  /// against measured on-device distances gives a threshold at which the gate
  /// rejects ~3% of synthetic non-signs (best Youden J across ALL thresholds:
  /// 0.027, where 0 is chance). It cannot separate the two populations here, so
  /// enforcing it only produces false rejections of correct predictions.
  ///
  /// Negative rejection is instead handled by the trained `none` class, which is
  /// what it was added for, and by [confidenceThreshold].
  ///
  /// The distance is still computed and reported -- it is useful telemetry, and
  /// cheap (a 32x32 quadratic form once per sign).
  LocalResult predict(
    List<Float32List> clip, {
    double? durationSec,
    int? frameWidth,
    int? frameHeight,
    bool enforceOodGate = false,
  }) {
    if (clip.length < 5) {
      throw ArgumentError('clip too short: ${clip.length} frames');
    }
    final started = DateTime.now();
    final rawFrames = clip.length;

    // Both corrections operate on RAW normalized landmarks, in this order, before
    // any of the nslr contract runs. Aspect first: it is a per-frame geometric
    // fix, and resampling afterwards then interpolates already-correct geometry.
    if (frameWidth != null && frameHeight != null) {
      correctAspect(clip, frameWidth, frameHeight);
    }
    var work = clip;
    var resampledTo = rawFrames;
    if (durationSec != null && durationSec > 0) {
      work = resampleToTrainingRate(clip, durationSec);
      resampledTo = work.length;
    }

    normalizeClip(work);
    final std = standardizeLength(work, seqLen);

    // tflite_flutter's documented path takes nested lists. Once per sign, so the
    // boxing cost is irrelevant next to the LSTM itself.
    final input = List.generate(
      1,
      (_) => List.generate(
        seqLen,
        (t) => List<double>.generate(
          Nslr.featureDim,
          (f) => std.data[t * Nslr.featureDim + f],
          growable: false,
        ),
        growable: false,
      ),
      growable: false,
    );

    final embDim = _embIndex >= 0
        ? _interpreter.getOutputTensor(_embIndex).shape.last
        : 0;
    final probsOut = [List<double>.filled(nClasses, 0)];
    final embOut = [List<double>.filled(embDim > 0 ? embDim : 1, 0)];

    final outputs = <int, Object>{_probsIndex: probsOut};
    if (_embIndex >= 0) outputs[_embIndex] = embOut;

    _interpreter.runForMultipleInputs([input], outputs);

    final probs = probsOut[0];
    final order = List<int>.generate(nClasses, (i) => i)
      ..sort((a, b) => probs[b].compareTo(probs[a]));

    final best = order.first;
    final bestKey = classNames[best];
    final confidence = probs[best];

    double? distance;
    var isUnknown = false;
    final gate = ood;
    if (gate != null && _embIndex >= 0) {
      // Always measured, only sometimes enforced.
      distance = gate.minDistance(embOut[0]);
      if (enforceOodGate) isUnknown = gate.rejects(distance);
    }

    // Decision rule. When the OOD gate is enforced it WINS over confidence -- a
    // clip can be 99% "call police" by softmax and still be unknown if its
    // embedding sits far from every prototype. With the gate off (the default on
    // device) only `none` and the confidence threshold can withhold an answer.
    //
    // The `none` class is folded into `unknown` here. It is a real trained class
    // (negatives: rest, random motion, partial signs), but treating it as an
    // `accepted` answer would make the app speak "Unknown / none of the above"
    // aloud at a signer — the live-demo hazard recorded as finding 8.1 in
    // ARCHITECTURE.md. The server still has this bug; this path does not.
    final String status;
    if (isUnknown || bestKey == 'none') {
      status = 'unknown';
    } else if (confidence >= confidenceThreshold) {
      status = 'accepted';
    } else {
      status = 'low_confidence';
    }

    return LocalResult(
      status: status,
      label: status == 'accepted' ? bestKey : null,
      display: status == 'accepted' ? (kClassLabels[bestKey] ?? bestKey) : null,
      confidence: confidence,
      bestGuess: bestKey,
      top3: order
          .take(math.min(3, nClasses))
          .map((i) => Guess(classNames[i], probs[i]))
          .toList(),
      oodDistance: distance,
      nFrames: rawFrames,
      framesUsed: resampledTo,
      standardize: std.mode,
      inferenceMs: DateTime.now().difference(started).inMilliseconds,
    );
  }

  void dispose() => _interpreter.close();
}
