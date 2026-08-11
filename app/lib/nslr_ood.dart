import 'dart:math' as math;
import 'dart:typed_data';

/// Dart port of `nslr/ood.py` — open-set rejection by Mahalanobis distance to
/// per-class prototypes.
///
/// The softmax is closed-world: it always names one of the trained phrases,
/// confidently, even for a gesture that is not a sign at all. This measures how
/// far an embedding sits from every class prototype in the penultimate Dense-32
/// space, so "that wasn't any sign I know" becomes expressible.
///
/// Kept as a separate module, like the Python side, so it can be tested against
/// golden values from the real fitted gate rather than only exercised through the
/// full interpreter.
class OodGate {
  OodGate({
    required this.means,
    required this.precision,
    required this.threshold,
  });

  /// Per-class prototypes, `k` rows of `d`.
  final List<Float64List> means;

  /// Shared within-class precision matrix, `d x d` (Ledoit-Wolf shrunk when it
  /// was fitted, which is why it is safe to invert at this sample size).
  final List<Float64List> precision;

  /// Distances above this are rejected. Fitted as the p99 of in-distribution
  /// training distances, so ~1% of real signs are expected to fall outside.
  final double threshold;

  int get nClasses => means.length;
  int get embeddingDim => means.isEmpty ? 0 : means.first.length;

  /// Parse the `ood.json` written by `scripts/export_tflite.py`.
  factory OodGate.fromJson(Map<String, dynamic> json) => OodGate(
        means: _toMatrix(json['means'] as List),
        precision: _toMatrix(json['precision'] as List),
        threshold: (json['threshold'] as num).toDouble(),
      );

  static List<Float64List> _toMatrix(List rows) => rows
      .map((r) => Float64List.fromList(
            (r as List).map((v) => (v as num).toDouble()).toList(),
          ))
      .toList();

  /// Smallest Mahalanobis distance from [z] to any prototype:
  /// `min_c sqrt((z - mu_c)^T P (z - mu_c))`.
  ///
  /// The quadratic form is evaluated directly rather than via a Cholesky factor —
  /// d is 32 and this runs once per sign, so 32x32 work is free and the code stays
  /// checkable against `mahalanobis_min`'s einsum.
  double minDistance(List<double> z) {
    if (means.isEmpty) return 0;
    final d = z.length;
    final diff = Float64List(d);
    var best = double.infinity;

    for (final mean in means) {
      for (var i = 0; i < d; i++) {
        diff[i] = z[i] - mean[i];
      }
      var acc = 0.0;
      for (var i = 0; i < d; i++) {
        final row = precision[i];
        var inner = 0.0;
        for (var j = 0; j < d; j++) {
          inner += row[j] * diff[j];
        }
        acc += diff[i] * inner;
      }
      // Clamp before the root: a shrunk precision matrix is positive definite in
      // theory, but floating point can still land a hair below zero when the
      // embedding sits exactly on a prototype.
      final dist = math.sqrt(acc < 0 ? 0 : acc);
      if (dist < best) best = dist;
    }
    return best;
  }

  bool rejects(double distance) => distance > threshold;
}
