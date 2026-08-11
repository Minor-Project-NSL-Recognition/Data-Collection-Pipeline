import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:nsl_app/nslr_ood.dart';
import 'package:nsl_app/nslr_preprocess.dart';

/// Proves the Dart port of the preprocessing contract equals the Python
/// pipeline, rather than assuming it.
///
/// This is the highest-risk surface in the offline path. A divergence here does
/// not throw — it feeds the BiLSTM features it was never trained on, and the
/// model answers confidently anyway. There is no symptom to notice at runtime,
/// which is exactly why it is pinned to golden values from the real pipeline.
///
/// Regenerate the fixture after any change to `nslr/preprocess.py`, `nslr/ood.py`
/// or the model:
///     python scripts/train_model.py
///     python scripts/export_tflite.py
///     python scripts/make_golden.py
///
/// The full end-to-end procedure is in RETRAIN.md at the repo root.
void main() {
  late Map<String, dynamic> golden;

  setUpAll(() {
    final file = File('test/fixtures/preprocess_golden.json');
    if (!file.existsSync()) {
      fail('missing ${file.path} — regenerate it (see the header comment)');
    }
    golden = jsonDecode(file.readAsStringSync()) as Map<String, dynamic>;
  });

  test('fixture matches the shipped model contract', () {
    expect(golden['feature_dim'], Nslr.featureDim);
    // seq_len has changed three times in this project (151 -> 137 -> 146). The
    // point of the test is that nothing in Dart hardcodes it.
    expect(golden['seq_len'], isA<int>());
    expect((golden['class_names'] as List).length, greaterThan(1));
  });

  group('normalizeClip + standardizeLength vs Python', () {
    test('reproduces every standardize branch exactly', () {
      final seqLen = golden['seq_len'] as int;
      final cases = golden['cases'] as List;
      expect(cases.length, greaterThanOrEqualTo(2),
          reason: 'need at least the padded and subsampled branches');

      final modesSeen = <String>{};

      for (final raw in cases) {
        final c = raw as Map<String, dynamic>;
        final name = c['name'] as String;
        final expectedMode = c['expected_mode'] as String;

        final clip = (c['raw'] as List)
            .map((row) => Float32List.fromList(
                  (row as List).map((v) => (v as num).toDouble()).toList(),
                ))
            .toList();
        expect(clip.length, c['n_frames'], reason: name);
        expect(clip.first.length, Nslr.featureDim, reason: name);

        normalizeClip(clip);
        final got = standardizeLength(clip, seqLen);
        modesSeen.add(got.mode);

        expect(got.mode, expectedMode, reason: name);
        expect(got.data.length, seqLen * Nslr.featureDim, reason: name);

        final expected = (c['expected'] as List)
            .expand((row) => (row as List).map((v) => (v as num).toDouble()))
            .toList();
        expect(expected.length, got.data.length, reason: name);

        // Tolerance, not equality: NumPy does the whole normalization in float32
        // while Dart computes in double and stores to float32, so the two differ
        // in the last bits. 1e-4 is far tighter than anything that could change a
        // prediction and far looser than that representation gap.
        var worst = 0.0;
        var worstAt = -1;
        for (var i = 0; i < expected.length; i++) {
          final diff = (got.data[i] - expected[i]).abs();
          if (diff > worst) {
            worst = diff;
            worstAt = i;
          }
        }
        expect(worst, lessThan(1e-4),
            reason: '$name: worst |delta| $worst at flat index $worstAt '
                '(frame ${worstAt ~/ Nslr.featureDim}, '
                'feature ${worstAt % Nslr.featureDim})');
      }

      expect(modesSeen, containsAll(<String>{'padded', 'subsampled'}),
          reason: 'the fixture must exercise both length branches');
    });

    test('padded frames are exactly zero, so Masking still skips them', () {
      final seqLen = golden['seq_len'] as int;
      final short = (golden['cases'] as List).cast<Map<String, dynamic>>().firstWhere(
            (c) => c['expected_mode'] == 'padded',
          );
      final n = short['n_frames'] as int;

      final clip = (short['raw'] as List)
          .map((row) => Float32List.fromList(
                (row as List).map((v) => (v as num).toDouble()).toList(),
              ))
          .toList();
      normalizeClip(clip);
      final got = standardizeLength(clip, seqLen);

      // Every feature of every padded timestep must be bit-zero. The exported
      // graph's Masking is a `NOT_EQUAL(0)` reduction, so a normalization that
      // turned padding into 1e-9 would silently un-mask the tail.
      for (var t = n; t < seqLen; t++) {
        for (var f = 0; f < Nslr.featureDim; f++) {
          expect(got.data[t * Nslr.featureDim + f], 0.0,
              reason: 'padded frame $t feature $f is non-zero');
        }
      }
    });

    test('an all-zero frame normalizes to all-zero', () {
      // The property the whole padding scheme rests on: the anchor formulas map
      // zero to zero because eps keeps the denominator finite. An undetected
      // block is all-zero, and must stay that way.
      final clip = [Float32List(Nslr.featureDim)];
      normalizeClip(clip);
      expect(clip.first.every((v) => v == 0.0), isTrue);
    });
  });

  group('OodGate vs Python mahalanobis_min', () {
    /// The gate the app actually ships, not a copy — so this also catches an
    /// `ood.json` asset that was never refreshed after a retrain.
    OodGate shippedGate() {
      final file = File('assets/models/ood.json');
      if (!file.existsSync()) {
        fail('missing ${file.path} — copy it from results/ after export_tflite.py');
      }
      return OodGate.fromJson(
          jsonDecode(file.readAsStringSync()) as Map<String, dynamic>);
    }

    test('the shipped gate agrees with the fixture it was fitted alongside', () {
      final ood = golden['ood'] as Map<String, dynamic>;
      final gate = shippedGate();
      expect(gate.threshold, closeTo((ood['threshold'] as num).toDouble(), 1e-9),
          reason: 'assets/models/ood.json is from a different run than the '
              'fixture — one of them is stale');
      expect(gate.nClasses, (golden['class_names'] as List).length);
    });

    test('reproduces the fitted gate distances', () {
      final ood = golden['ood'] as Map<String, dynamic>;
      final gate = shippedGate();

      final embeddings = (ood['embeddings'] as List)
          .map((row) => (row as List).map((v) => (v as num).toDouble()).toList())
          .toList();
      final expected = (ood['expected_distances'] as List)
          .map((v) => (v as num).toDouble())
          .toList();

      expect(embeddings.length, expected.length);
      expect(gate.embeddingDim, embeddings.first.length);

      for (var i = 0; i < embeddings.length; i++) {
        final got = gate.minDistance(embeddings[i]);
        // Both sides are float64 here, so this can be tight. A transposed
        // precision matrix or a sign slip would blow past this by orders of
        // magnitude.
        expect((got - expected[i]).abs(), lessThan(1e-6),
            reason: 'embedding $i: got $got want ${expected[i]}');
      }
    });

    test('a prototype sits at distance zero', () {
      final ood = golden['ood'] as Map<String, dynamic>;
      final expected = (ood['expected_distances'] as List)
          .map((v) => (v as num).toDouble())
          .toList();
      // The fixture alternates prototype / displaced, so the even entries are
      // exact prototypes and must be ~0 rather than merely small.
      expect(expected[0], lessThan(1e-9));
    });
  });
}
