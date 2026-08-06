import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

/// Converts camera frames to upright JPEGs on a background isolate.
///
/// This is the client's bottleneck, not the network. Colour conversion plus
/// JPEG encoding of a 640x480 frame costs roughly 50-120 ms in pure Dart, so
/// doing it on the UI isolate would freeze the preview. A *persistent* isolate
/// matters too: `compute()` spawns a fresh one per call, and at ~16 fps that
/// overhead dominates the work itself.
///
/// Backpressure is deliberate — [encode] returns null while a frame is already
/// in flight rather than queueing. Dropping frames is correct here: the server
/// decimates to ~15.7 fps anyway, and a queue would only add latency.
class FrameEncoder {
  FrameEncoder._(this._isolate, this._toIsolate, this._fromIsolate);

  final Isolate _isolate;
  final SendPort _toIsolate;
  final ReceivePort _fromIsolate;
  Completer<Uint8List?>? _inFlight;

  bool get isBusy => _inFlight != null;

  static Future<FrameEncoder> spawn() async {
    final rp = ReceivePort();
    final isolate = await Isolate.spawn(_entry, rp.sendPort);
    final ready = Completer<SendPort>();
    late final StreamSubscription sub;
    FrameEncoder? self;

    sub = rp.listen((msg) {
      if (msg is SendPort) {
        ready.complete(msg);
      } else {
        self?._inFlight?.complete(msg as Uint8List?);
        self?._inFlight = null;
      }
    });
    final sendPort = await ready.future;
    self = FrameEncoder._(isolate, sendPort, rp);
    // keep the subscription alive for the encoder's lifetime
    self._sub = sub;
    return self;
  }

  StreamSubscription? _sub;

  /// [rotation] is the camera's `sensorOrientation` in degrees. Android
  /// delivers frames in sensor orientation, so a portrait-locked app must
  /// rotate them or MediaPipe sees a person lying on their side and detects
  /// no pose at all.
  ///
  /// [maxWidth] downscales before encoding. It is the main performance lever,
  /// but lowering it shrinks the hands in frame, which is where landmark
  /// quality degrades first — prefer 480 or above.
  Future<Uint8List?> encode(
    CameraImage image, {
    required int rotation,
    int quality = 75,
    int maxWidth = 640,
  }) {
    if (_inFlight != null) return Future.value(null); // drop, don't queue
    final completer = Completer<Uint8List?>();
    _inFlight = completer;
    _toIsolate.send(_Job(
      width: image.width,
      height: image.height,
      format: image.format.group == ImageFormatGroup.bgra8888 ? 'bgra' : 'yuv420',
      planes: image.planes
          .map((p) => _Plane(p.bytes, p.bytesPerRow, p.bytesPerPixel ?? 1))
          .toList(),
      rotation: rotation,
      quality: quality,
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
    required this.quality,
    required this.maxWidth,
  });
  final int width;
  final int height;
  final String format;
  final List<_Plane> planes;
  final int rotation;
  final int quality;
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

Uint8List? _convert(_Job job) {
  final w = job.width, h = job.height;
  // Fill a flat RGB buffer and hand it to `Image.fromBytes` in one go.
  // Per-pixel setPixelRgb() on an Image is several times slower.
  final rgb = Uint8List(w * h * 3);

  if (job.format == 'bgra') {
    final src = job.planes[0].bytes;
    final stride = job.planes[0].bytesPerRow;
    for (var y = 0; y < h; y++) {
      var si = y * stride;
      var di = y * w * 3;
      for (var x = 0; x < w; x++) {
        rgb[di] = src[si + 2]; // R
        rgb[di + 1] = src[si + 1]; // G
        rgb[di + 2] = src[si]; // B
        si += 4;
        di += 3;
      }
    }
  } else {
    final yP = job.planes[0], uP = job.planes[1], vP = job.planes[2];
    final yB = yP.bytes, uB = uP.bytes, vB = vP.bytes;
    final yStride = yP.bytesPerRow;
    final uvStride = uP.bytesPerRow;
    final uvPixel = uP.bytesPerPixel;

    for (var y = 0; y < h; y++) {
      final yRow = y * yStride;
      final uvRow = (y >> 1) * uvStride;
      var di = y * w * 3;
      for (var x = 0; x < w; x++) {
        final uvIdx = uvRow + (x >> 1) * uvPixel;
        final yv = yB[yRow + x];
        final u = uB[uvIdx] - 128;
        final v = vB[uvIdx] - 128;
        // BT.601 full-range, matching what cv2.cvtColor produces server-side.
        var r = yv + 1.370705 * v;
        var g = yv - 0.337633 * u - 0.698001 * v;
        var b = yv + 1.732446 * u;
        rgb[di] = r < 0 ? 0 : (r > 255 ? 255 : r.toInt());
        rgb[di + 1] = g < 0 ? 0 : (g > 255 ? 255 : g.toInt());
        rgb[di + 2] = b < 0 ? 0 : (b > 255 ? 255 : b.toInt());
        di += 3;
      }
    }
  }

  var frame = img.Image.fromBytes(
    width: w,
    height: h,
    bytes: rgb.buffer,
    numChannels: 3,
  );

  if (job.rotation % 360 != 0) {
    frame = img.copyRotate(frame, angle: job.rotation);
  }
  if (frame.width > job.maxWidth) {
    frame = img.copyResize(frame, width: job.maxWidth);
  }
  return img.encodeJpg(frame, quality: job.quality);
}
