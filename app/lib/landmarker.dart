import 'dart:typed_data';

import 'package:flutter/services.dart';

import 'nslr_preprocess.dart';

/// One extracted frame: the 225-vector plus what the detectors actually found.
class LandmarkFrame {
  LandmarkFrame({
    required this.vector,
    required this.pose,
    required this.leftHand,
    required this.rightHand,
    this.handednessAgrees,
  });

  final Float32List vector;
  final bool pose;
  final bool leftHand;
  final bool rightHand;

  /// Whether MediaPipe's handedness classifier agreed with the nearest-pose-wrist
  /// assignment on this frame. Null when there was nothing to compare. Carried
  /// through for the same reason `parity_report.py` reports it: if this drops,
  /// the left/right hand blocks may be swapping and nothing else would say so.
  final bool? handednessAgrees;

  bool get anyHand => leftHand || rightHand;
}

/// Dart side of the on-device MediaPipe bridge (`LandmarkerBridge.kt`).
///
/// Replaces the WebSocket to the Python server: same 225-vector contract, no
/// network. The `.task` bundles are read from Flutter assets by the native side,
/// which is why [init] passes asset *lookup keys* rather than file paths.
class Landmarker {
  static const _channel = MethodChannel('np.edu.nsl.nsl_app/landmarker');

  static const poseAsset = 'assets/models/pose_landmarker_full.task';
  static const handAsset = 'assets/models/hand_landmarker.task';

  bool _ready = false;
  bool get isReady => _ready;

  /// Loads both landmarkers. Tries the GPU delegate and falls back to CPU
  /// natively, so this succeeds on emulators too.
  ///
  /// Slow (hundreds of ms — it reads 17 MB of model bundles), so call it once at
  /// page start, not per clip.
  Future<void> init({bool useGpu = true}) async {
    await _channel.invokeMethod<void>('init', {
      // Flutter packages assets under a prefix that is Flutter's business, so
      // resolve the real key rather than hardcoding it.
      'poseAsset': _lookupKey(poseAsset),
      'handAsset': _lookupKey(handAsset),
      'useGpu': useGpu,
    });
    _ready = true;
  }

  /// Flutter's asset lookup key for a pubspec asset path.
  ///
  /// Android packages Flutter assets under `flutter_assets/`. There is no public
  /// Dart API for this mapping, but it has been stable for years and the native
  /// side reads it straight from the AssetManager.
  String _lookupKey(String assetPath) => 'flutter_assets/$assetPath';

  /// Extract one frame. Returns null when the native side dropped it because the
  /// previous frame was still in the detectors — that is normal backpressure, not
  /// an error.
  ///
  /// [timestampMs] must increase across a clip; the native side clamps rather
  /// than throwing if it doesn't.
  Future<LandmarkFrame?> detect({
    required Uint8List rgba,
    required int width,
    required int height,
    required int timestampMs,
  }) async {
    final res = await _channel.invokeMapMethod<String, dynamic>('detect', {
      'rgba': rgba,
      'width': width,
      'height': height,
      'timestampMs': timestampMs,
    });
    if (res == null || res['dropped'] == true) return null;

    // The native side sends double[] (see LandmarkerBridge.assemble). Accept any
    // numeric list rather than one exact type: the codec's choice of typed-data
    // class is an implementation detail, and a mismatch here would silently drop
    // every single frame.
    final raw = res['vector'];
    final List<double> values;
    if (raw is Float64List) {
      values = raw;
    } else if (raw is Float32List) {
      values = raw;
    } else if (raw is List) {
      values = raw.map((v) => (v as num).toDouble()).toList();
    } else {
      throw StateError('vector had unexpected type ${raw.runtimeType}');
    }
    if (values.length != Nslr.featureDim) {
      throw StateError('vector length ${values.length} != ${Nslr.featureDim}');
    }

    return LandmarkFrame(
      // Build our OWN buffer rather than passing a decoded view through.
      //
      // The codec decodes typed data as a view onto the platform message buffer,
      // which the engine owns and marks read-only. Handing that to
      // `normalizeClip` — which normalizes in place — throws "Unsupported
      // operation: Cannot modify an unmodifiable list" at the moment the user
      // stops recording, after a whole sign has already been captured.
      vector: Float32List.fromList(values),
      pose: res['pose'] == true,
      leftHand: res['leftHand'] == true,
      rightHand: res['rightHand'] == true,
      handednessAgrees: res['handednessAgrees'] as bool?,
    );
  }

  /// Start a new clip's timestamp sequence. The detectors keep their tracking
  /// state, matching `record.py` holding one Holistic graph open across clips.
  Future<void> reset() => _channel.invokeMethod<void>('reset');

  /// Write a raw clip to external files and return its on-device path.
  ///
  /// Diagnostics only. `record.py` saved landmarks so they could be inspected
  /// later; the phone never did, which is why on-device disagreement with the
  /// Python pipeline has been so hard to pin down. Pull one with:
  ///
  ///   adb pull /sdcard/Android/data/np.edu.nsl.nsl_app/files/clips
  ///   python scripts/inspect_device_clip.py `clips/<file>.f32`
  Future<String?> saveClip(List<Float32List> clip, String name) async {
    final flat = Float32List(clip.length * Nslr.featureDim);
    for (var t = 0; t < clip.length; t++) {
      flat.setRange(t * Nslr.featureDim, (t + 1) * Nslr.featureDim, clip[t]);
    }
    return _channel.invokeMethod<String>('saveClip', {
      // Uint8List view over the same bytes — no copy, and the codec sends it as
      // a byte array, which every Flutter version encodes identically.
      'bytes': flat.buffer.asUint8List(flat.offsetInBytes, flat.lengthInBytes),
      'name': name,
    });
  }

  Future<void> close() async {
    if (!_ready) return;
    _ready = false;
    await _channel.invokeMethod<void>('close');
  }
}
