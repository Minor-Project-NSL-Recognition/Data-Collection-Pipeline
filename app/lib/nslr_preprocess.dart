import 'dart:math' as math;
import 'dart:typed_data';

/// Dart port of `nslr/config.py` + `nslr/preprocess.py`.
///
/// These constants and formulas are the contract between the training pipeline
/// and every inference path. If this file and `nslr/preprocess.py` ever disagree,
/// the model receives features it was never trained on and the failure is silent
/// — wrong answers with high confidence, not an exception. Change both or
/// neither.
class Nslr {
  static const int poseLandmarks = 33;
  static const int handLandmarks = 21;
  static const int poseDim = poseLandmarks * 3; // 99
  static const int handDim = handLandmarks * 3; // 63
  static const int featureDim = poseDim + handDim * 2; // 225

  /// Offsets of the three blocks in the 225-vector, in `record.py`'s
  /// concatenation order.
  static const int poseOffset = 0;
  static const int leftHandOffset = poseDim;
  static const int rightHandOffset = poseDim + handDim;

  /// Normalization anchors (MediaPipe landmark indices).
  static const int poseLeftShoulder = 11;
  static const int poseRightShoulder = 12;
  static const int handWrist = 0;
  static const int handMiddleFingerMcp = 9;

  static const double eps = 1e-6;
}

/// `(p_i - mid_shoulders) / (shoulder_width + eps)`, in place over one frame's
/// pose block.
///
/// An all-zero block maps to all-zero: the anchors are themselves zero, so the
/// numerator is zero while `eps` keeps the denominator finite. That property is
/// what lets padded frames stay exactly zero and remain maskable — it is load
/// bearing, not a happy accident.
void _normalizePoseBlock(Float32List frame, int offset) {
  final lx = frame[offset + Nslr.poseLeftShoulder * 3];
  final ly = frame[offset + Nslr.poseLeftShoulder * 3 + 1];
  final lz = frame[offset + Nslr.poseLeftShoulder * 3 + 2];
  final rx = frame[offset + Nslr.poseRightShoulder * 3];
  final ry = frame[offset + Nslr.poseRightShoulder * 3 + 1];
  final rz = frame[offset + Nslr.poseRightShoulder * 3 + 2];

  final midX = 0.5 * (lx + rx);
  final midY = 0.5 * (ly + ry);
  final midZ = 0.5 * (lz + rz);

  final dx = lx - rx, dy = ly - ry, dz = lz - rz;
  final scale = math.sqrt(dx * dx + dy * dy + dz * dz) + Nslr.eps;

  for (var i = 0; i < Nslr.poseLandmarks; i++) {
    final b = offset + i * 3;
    frame[b] = (frame[b] - midX) / scale;
    frame[b + 1] = (frame[b + 1] - midY) / scale;
    frame[b + 2] = (frame[b + 2] - midZ) / scale;
  }
}

/// `(h_i - h_0) / (||h_9 - h_0|| + eps)`, in place over one hand block.
void _normalizeHandBlock(Float32List frame, int offset) {
  final wx = frame[offset + Nslr.handWrist * 3];
  final wy = frame[offset + Nslr.handWrist * 3 + 1];
  final wz = frame[offset + Nslr.handWrist * 3 + 2];
  final mx = frame[offset + Nslr.handMiddleFingerMcp * 3];
  final my = frame[offset + Nslr.handMiddleFingerMcp * 3 + 1];
  final mz = frame[offset + Nslr.handMiddleFingerMcp * 3 + 2];

  final dx = mx - wx, dy = my - wy, dz = mz - wz;
  final scale = math.sqrt(dx * dx + dy * dy + dz * dz) + Nslr.eps;

  for (var i = 0; i < Nslr.handLandmarks; i++) {
    final b = offset + i * 3;
    frame[b] = (frame[b] - wx) / scale;
    frame[b + 1] = (frame[b + 1] - wy) / scale;
    frame[b + 2] = (frame[b + 2] - wz) / scale;
  }
}

/// Normalize every frame of a raw clip, blocks independent. Mutates in place —
/// the caller owns freshly-built vectors and nothing else reads them.
void normalizeClip(List<Float32List> clip) {
  for (final frame in clip) {
    _normalizePoseBlock(frame, Nslr.poseOffset);
    _normalizeHandBlock(frame, Nslr.leftHandOffset);
    _normalizeHandBlock(frame, Nslr.rightHandOffset);
  }
}

/// NumPy's `ndarray.round()` — half-to-even, not half-away-from-zero.
///
/// Dart's `double.round()` rounds 0.5 up; NumPy rounds it to the nearest even
/// integer. `standardizeLength` feeds `linspace(...).round()` straight into an
/// index, so on an exact .5 the two conventions pick different frames. Rare, and
/// harmless when it happens, but this path is supposed to be bit-comparable with
/// the Python pipeline and a silent off-by-one index is exactly the kind of drift
/// that is impossible to find later.
int _roundHalfToEven(double v) {
  final floor = v.floorToDouble();
  final diff = v - floor;
  if (diff > 0.5) return floor.toInt() + 1;
  if (diff < 0.5) return floor.toInt();
  final f = floor.toInt();
  return f.isEven ? f : f + 1;
}

/// Result of length standardization: a flat `seqLen * 225` tensor ready for the
/// interpreter, plus which branch produced it (useful telemetry — `subsampled`
/// means the clip was long enough that its duration signal was rescaled away).
class Standardized {
  Standardized(this.data, this.mode, this.realFrames);
  final Float32List data;
  final String mode;
  final int realFrames;
}

/// Over-length -> uniform temporal subsampling spanning the whole clip;
/// under-length -> zero-pad at the end. Mirrors `standardize_length()`.
///
/// No mask is produced or needed: the `Masking` layer is baked into the exported
/// graph (it shows up as NOT_EQUAL/REDUCE_ANY/SELECT_V2 ops), so zero-padded
/// frames are skipped by the model itself.
Standardized standardizeLength(List<Float32List> clip, int seqLen) {
  final out = Float32List(seqLen * Nslr.featureDim);
  final n = clip.length;

  if (n == seqLen) {
    for (var t = 0; t < seqLen; t++) {
      out.setRange(t * Nslr.featureDim, (t + 1) * Nslr.featureDim, clip[t]);
    }
    return Standardized(out, 'exact', n);
  }

  if (n > seqLen) {
    // np.linspace(0, n - 1, seqLen).round().astype(int)
    for (var t = 0; t < seqLen; t++) {
      final pos = (n - 1) * t / (seqLen - 1);
      var idx = _roundHalfToEven(pos);
      if (idx < 0) idx = 0;
      if (idx > n - 1) idx = n - 1;
      out.setRange(t * Nslr.featureDim, (t + 1) * Nslr.featureDim, clip[idx]);
    }
    return Standardized(out, 'subsampled', n);
  }

  for (var t = 0; t < n; t++) {
    out.setRange(t * Nslr.featureDim, (t + 1) * Nslr.featureDim, clip[t]);
  }
  // The tail is already zero — Float32List is zero-initialised.
  return Standardized(out, 'padded', n);
}
