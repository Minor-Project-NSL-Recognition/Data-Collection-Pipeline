import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:camera/camera.dart';

/// Camera frame -> upright, mirrored RGBA bytes, on a background isolate.
///
/// This is [FrameEncoder] with the JPEG step removed. The offline path hands
/// pixels straight to MediaPipe, so encoding a JPEG only to have native code
/// decode it again would be pure waste — and JPEG encoding in pure Dart was the
/// old client's single biggest cost (~40 ms/frame at 320 px).
///
/// The rotation and YUV maths are deliberately identical to [FrameEncoder]'s:
/// they were validated against the server's `cv2` path, and the landmarks the
/// model consumes are sensitive to both.
///
/// **Output is RGBA, not RGB**, so Kotlin can hand it to
/// `Bitmap.copyPixelsFromBuffer` on an `ARGB_8888` bitmap with no repacking —
/// that config's in-memory byte order is R,G,B,A.
///
/// **Output is mirrored.** `record.py` ran `cv2.flip(frame, 1)` before detection,
/// so every training landmark describes a mirrored image, and the server
/// reproduces that flip. Getting this wrong does not fail loudly — it silently
/// swaps the left and right hand blocks of the 225-vector.
class RgbaConverter {
  RgbaConverter._(this._isolate, this._toIsolate, this._fromIsolate);

  final Isolate _isolate;
  final SendPort _toIsolate;
  final ReceivePort _fromIsolate;
  Completer<RgbaFrame?>? _inFlight;
  StreamSubscription<dynamic>? _sub;

  bool get isBusy => _inFlight != null;

  static Future<RgbaConverter> spawn() async {
    final rp = ReceivePort();
    final isolate = await Isolate.spawn(_entry, rp.sendPort);
    final ready = Completer<SendPort>();
    RgbaConverter? self;

    final sub = rp.listen((msg) {
      if (msg is SendPort) {
        ready.complete(msg);
      } else {
        final pending = self?._inFlight;
        self?._inFlight = null;
        pending?.complete(msg as RgbaFrame?);
      }
    });

    final sendPort = await ready.future;
    self = RgbaConverter._(isolate, sendPort, rp);
    self._sub = sub;
    return self;
  }

  /// [rotation] is the camera's `sensorOrientation` in degrees. Android delivers
  /// frames in sensor orientation; a portrait-locked app must rotate them or
  /// MediaPipe sees a person lying on their side and finds no pose at all.
  ///
  /// [maxWidth] caps the width of the upright frame. Larger than the old JPEG
  /// path's 320 because we no longer pay for encoding, and hand detection is the
  /// first thing to suffer when the hands get small in frame — the standalone
  /// HandLandmarker has no pose ROI to fall back on.
  Future<RgbaFrame?> convert(
    CameraImage image, {
    required int rotation,
    bool mirror = true,
    int maxWidth = 480,
  }) {
    if (_inFlight != null) return Future.value(null); // drop, don't queue
    final completer = Completer<RgbaFrame?>();
    _inFlight = completer;
    _toIsolate.send(_Job(
      width: image.width,
      height: image.height,
      format: image.format.group == ImageFormatGroup.bgra8888 ? 'bgra' : 'yuv420',
      planes: image.planes
          .map((p) => _Plane(p.bytes, p.bytesPerRow, p.bytesPerPixel ?? 1))
          .toList(),
      rotation: rotation,
      mirror: mirror,
      maxWidth: maxWidth,
    ));
    return completer.future;
  }

  void dispose() {
    _sub?.cancel();
    _fromIsolate.close();
    _isolate.kill(priority: Isolate.immediate);
  }
}

class RgbaFrame {
  RgbaFrame({
    required this.bytes,
    required this.width,
    required this.height,
    required this.millis,
    required this.srcWidth,
    required this.srcHeight,
  });

  final Uint8List bytes;
  final int width;
  final int height;
  final int millis;
  final int srcWidth;
  final int srcHeight;
}

class _Plane {
  _Plane(this.bytes, this.bytesPerRow, this.bytesPerPixel);
  final Uint8List bytes;
  final int bytesPerRow;
  final int bytesPerPixel;
}

class _Job {
  _Job({
    required this.width,
    required this.height,
    required this.format,
    required this.planes,
    required this.rotation,
    required this.mirror,
    required this.maxWidth,
  });
  final int width;
  final int height;
  final String format;
  final List<_Plane> planes;
  final int rotation;
  final bool mirror;
  final int maxWidth;
}

void _entry(SendPort toMain) {
  final rp = ReceivePort();
  toMain.send(rp.sendPort);
  rp.listen((msg) {
    if (msg is! _Job) return;
    try {
      toMain.send(_convert(msg));
    } catch (_) {
      toMain.send(null);
    }
  });
}

RgbaFrame? _convert(_Job job) {
  final started = DateTime.now();
  final w = job.width, h = job.height;

  final rot = ((job.rotation % 360) + 360) % 360;
  final swap = rot == 90 || rot == 270;

  // Dimensions once upright, before any downscale.
  final rw = swap ? h : w;
  final rh = swap ? w : h;

  var outW = rw, outH = rh;
  if (job.maxWidth > 0 && rw > job.maxWidth) {
    outW = job.maxWidth;
    outH = (rh * job.maxWidth / rw).round();
  }
  if (outW < 1 || outH < 1) return null;

  final rgba = Uint8List(outW * outH * 4);
  var di = 0;

  if (job.format == 'bgra') {
    final src = job.planes[0].bytes;
    final stride = job.planes[0].bytesPerRow;
    for (var oy = 0; oy < outH; oy++) {
      final ry = oy * rh ~/ outH;
      for (var ox = 0; ox < outW; ox++) {
        // The mirror is applied here, in the OUTPUT column index, so it costs
        // nothing and cannot be forgotten downstream.
        final mx = job.mirror ? (outW - 1 - ox) : ox;
        final rx = mx * rw ~/ outW;
        final int sx, sy;
        switch (rot) {
          case 90:
            sx = ry;
            sy = h - 1 - rx;
          case 180:
            sx = w - 1 - rx;
            sy = h - 1 - ry;
          case 270:
            sx = w - 1 - ry;
            sy = rx;
          default:
            sx = rx;
            sy = ry;
        }
        final si = sy * stride + sx * 4;
        rgba[di] = src[si + 2]; // R
        rgba[di + 1] = src[si + 1]; // G
        rgba[di + 2] = src[si]; // B
        rgba[di + 3] = 255;
        di += 4;
      }
    }
  } else {
    final yP = job.planes[0], uP = job.planes[1], vP = job.planes[2];
    final yB = yP.bytes, uB = uP.bytes, vB = vP.bytes;
    final yStride = yP.bytesPerRow;
    final uvStride = uP.bytesPerRow;
    final uvPixel = uP.bytesPerPixel;

    for (var oy = 0; oy < outH; oy++) {
      final ry = oy * rh ~/ outH;
      for (var ox = 0; ox < outW; ox++) {
        final mx = job.mirror ? (outW - 1 - ox) : ox;
        final rx = mx * rw ~/ outW;
        final int sx, sy;
        switch (rot) {
          case 90:
            sx = ry;
            sy = h - 1 - rx;
          case 180:
            sx = w - 1 - rx;
            sy = h - 1 - ry;
          case 270:
            sx = w - 1 - ry;
            sy = rx;
          default:
            sx = rx;
            sy = ry;
        }

        final uvIdx = (sy >> 1) * uvStride + (sx >> 1) * uvPixel;
        final yv = yB[yStride * sy + sx];
        final u = uB[uvIdx] - 128;
        final v = vB[uvIdx] - 128;
        // BT.601 full-range in 10-bit fixed point — the same constants the JPEG
        // path used, which were matched against the server's cv2.cvtColor.
        final r = (yv * 1024 + 1403 * v) >> 10;
        final g = (yv * 1024 - 346 * u - 715 * v) >> 10;
        final b = (yv * 1024 + 1774 * u) >> 10;
        rgba[di] = r < 0 ? 0 : (r > 255 ? 255 : r);
        rgba[di + 1] = g < 0 ? 0 : (g > 255 ? 255 : g);
        rgba[di + 2] = b < 0 ? 0 : (b > 255 ? 255 : b);
        rgba[di + 3] = 255;
        di += 4;
      }
    }
  }

  return RgbaFrame(
    bytes: rgba,
    width: outW,
    height: outH,
    millis: DateTime.now().difference(started).inMilliseconds,
    srcWidth: w,
    srcHeight: h,
  );
}
