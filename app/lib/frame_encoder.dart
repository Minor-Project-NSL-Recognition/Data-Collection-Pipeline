import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

/// Converts camera frames to upright JPEGs on a background isolate.
///
/// This is the client's bottleneck, not the network, and JPEG encoding in pure
/// Dart is most of it. Everything here exists to keep the pixel count fed to
/// [img.encodeJpg] as small as it can be without hurting hand landmarks:
///
///  * Rotation and downscaling are folded into the colour-conversion loop, so
///    there is one pass over the OUTPUT pixels and no intermediate full-size
///    buffers. The previous version converted the full frame, rotated it with
///    `copyRotate`, then resized -- three allocations and three passes, the
///    first two at full resolution.
///  * Sampling is nearest-neighbour, so a 720p source costs the same as a 480p
///    one once we are downscaling. Only [maxWidth] drives the cost.
///  * Integer fixed-point YUV maths, because the per-pixel `toInt()` on doubles
///    was measurable at this call count.
///
/// A *persistent* isolate matters too: `compute()` spawns a fresh one per call,
/// and at ~16 fps that overhead dominates the work itself.
///
/// Backpressure is deliberate -- [encode] returns null while a frame is already
/// in flight rather than queueing. Dropping frames is correct here: the server
/// decimates to ~15.7 fps anyway, and a queue would only add latency.
class FrameEncoder {
  FrameEncoder._(this._isolate, this._toIsolate, this._fromIsolate);

  final Isolate _isolate;
  final SendPort _toIsolate;
  final ReceivePort _fromIsolate;
  Completer<EncodedFrame?>? _inFlight;

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
        self?._inFlight?.complete(msg as EncodedFrame?);
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
  /// [maxWidth] caps the width of the *upright* frame -- the short side, since
  /// the result is portrait. It is the one performance lever that matters:
  /// encoding cost is roughly linear in output pixels. Lowering it shrinks the
  /// hands in frame, which is where landmark quality degrades first.
  Future<EncodedFrame?> encode(
    CameraImage image, {
    required int rotation,
    int quality = 65,
    int maxWidth = 320,
  }) {
    if (_inFlight != null) return Future.value(null); // drop, don't queue
    final completer = Completer<EncodedFrame?>();
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

/// The JPEG plus what it cost, so the UI can show where the frame budget went
/// instead of leaving a slow device as a mystery.
class EncodedFrame {
  EncodedFrame({
    required this.bytes,
    required this.millis,
    required this.srcWidth,
    required this.srcHeight,
    required this.outWidth,
    required this.outHeight,
  });

  final Uint8List bytes;
  final int millis;
  final int srcWidth;
  final int srcHeight;
  final int outWidth;
  final int outHeight;
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

EncodedFrame? _convert(_Job job) {
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

  final rgb = Uint8List(outW * outH * 3);
  var di = 0;

  if (job.format == 'bgra') {
    final src = job.planes[0].bytes;
    final stride = job.planes[0].bytesPerRow;
    for (var oy = 0; oy < outH; oy++) {
      final ry = oy * rh ~/ outH;
      for (var ox = 0; ox < outW; ox++) {
        final rx = ox * rw ~/ outW;
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
        rgb[di] = src[si + 2]; // R
        rgb[di + 1] = src[si + 1]; // G
        rgb[di + 2] = src[si]; // B
        di += 3;
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
        final rx = ox * rw ~/ outW;
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
        // BT.601 full-range in 10-bit fixed point, matching cv2.cvtColor
        // server-side to within a rounding step.
        final r = (yv * 1024 + 1403 * v) >> 10;
        final g = (yv * 1024 - 346 * u - 715 * v) >> 10;
        final b = (yv * 1024 + 1774 * u) >> 10;
        rgb[di] = r < 0 ? 0 : (r > 255 ? 255 : r);
        rgb[di + 1] = g < 0 ? 0 : (g > 255 ? 255 : g);
        rgb[di + 2] = b < 0 ? 0 : (b > 255 ? 255 : b);
        di += 3;
      }
    }
  }

  final frame = img.Image.fromBytes(
    width: outW,
    height: outH,
    bytes: rgb.buffer,
    numChannels: 3,
  );
  // 4:2:0 quarters the chroma the encoder has to transform and entropy-code.
  // The package default is 4:4:4, which is unusual for photographic JPEG and
  // was costing ~a third of the encode for detail MediaPipe does not use --
  // landmarks come from luma structure, not chroma resolution.
  final jpeg = img.encodeJpg(
    frame,
    quality: job.quality,
    chroma: img.JpegChroma.yuv420,
  );

  return EncodedFrame(
    bytes: jpeg,
    millis: DateTime.now().difference(started).inMilliseconds,
    srcWidth: w,
    srcHeight: h,
    outWidth: outW,
    outHeight: outH,
  );
}
