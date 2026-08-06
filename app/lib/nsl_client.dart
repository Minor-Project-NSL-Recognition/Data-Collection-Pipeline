import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:web_socket_channel/web_socket_channel.dart';

/// Client for the server's `WS /ws/stream` protocol (see server/README.md).
///
/// Frames are streamed *while the user signs* rather than uploaded afterwards.
/// Landmark extraction costs ~38 ms/frame on the server and is the entire
/// pipeline cost, so sending live hides it behind the sign itself: by the time
/// [finish] is called the server has already consumed the clip and only needs
/// ~15 ms to answer.

/// Per-frame acknowledgement. [pose] and [hands] are the server's live
/// detection flags — the cheapest possible check that the camera is framed and
/// oriented correctly, since a wrongly-rotated frame shows pose = false.
class NslAck {
  const NslAck({
    required this.accepted,
    required this.nFrames,
    this.pose = false,
    this.hands = false,
    this.reason,
  });

  final bool accepted;
  final int nFrames;
  final bool pose;
  final bool hands;

  /// Why the frame was skipped: `rate_limited` (normal and expected — the
  /// server decimates to ~15.7 fps), `decode_failed`, `max_frames`.
  final String? reason;

  factory NslAck.fromJson(Map<String, dynamic> j) => NslAck(
        accepted: j['accepted'] as bool? ?? false,
        nFrames: j['n_frames'] as int? ?? 0,
        pose: j['pose'] as bool? ?? false,
        hands: j['hands'] as bool? ?? false,
        reason: j['reason'] as String?,
      );
}

class NslGuess {
  const NslGuess(this.label, this.confidence);
  final String label;
  final double confidence;
}

class NslResult {
  const NslResult({
    required this.status,
    required this.confidence,
    required this.bestGuess,
    required this.top3,
    required this.nFrames,
    this.label,
    this.display,
    this.oodDistance,
    this.stats,
  });

  /// `accepted` | `low_confidence` | `unknown`.
  ///
  /// Only `accepted` should be spoken aloud. The softmax is closed-world and
  /// will confidently name a phrase for any input whatsoever; `unknown` is the
  /// open-set gate catching inputs that sit far from every known sign.
  final String status;
  final double confidence;
  final String bestGuess;
  final List<NslGuess> top3;
  final int nFrames;
  final String? label;
  final String? display;
  final double? oodDistance;
  final Map<String, dynamic>? stats;

  bool get isAccepted => status == 'accepted';

  factory NslResult.fromJson(Map<String, dynamic> j) => NslResult(
        status: j['status'] as String? ?? 'unknown',
        confidence: (j['confidence'] as num?)?.toDouble() ?? 0,
        bestGuess: j['best_guess'] as String? ?? '?',
        label: j['label'] as String?,
        display: j['display'] as String?,
        oodDistance: (j['ood_distance'] as num?)?.toDouble(),
        nFrames: j['n_frames'] as int? ?? 0,
        stats: j['stats'] as Map<String, dynamic>?,
        top3: ((j['top3'] as List?) ?? [])
            .map((e) => NslGuess(
                  e['label'] as String,
                  (e['confidence'] as num).toDouble(),
                ))
            .toList(),
      );
}

class NslException implements Exception {
  NslException(this.code, [this.detail]);
  final String code;
  final String? detail;
  @override
  String toString() => detail == null ? code : '$code: $detail';
}

class NslClient {
  WebSocketChannel? _channel;
  StreamSubscription? _sub;
  final _ackController = StreamController<NslAck>.broadcast();
  Completer<NslResult>? _pending;
  Completer<void>? _ready;
  Completer<void>? _resetDone;

  Stream<NslAck> get acks => _ackController.stream;
  bool get isConnected => _channel != null;

  /// [baseUrl] is the plain HTTP origin, e.g. `http://192.168.1.7:8000`.
  ///
  /// [mirror] must stay true for a front camera: the server flips the frame to
  /// match `record.py`, which mirrored every training clip. Flipping twice puts
  /// the left and right hand blocks in each other's slots.
  Future<void> connect(String baseUrl, {bool mirror = true, Duration timeout = const Duration(seconds: 10)}) async {
    await dispose();
    final ws = baseUrl.replaceFirst(RegExp(r'^http'), 'ws');
    final uri = Uri.parse('$ws/ws/stream?mirror=${mirror ? 1 : 0}');

    final channel = WebSocketChannel.connect(uri);
    _channel = channel;
    _ready = Completer<void>();

    _sub = channel.stream.listen(
      _onMessage,
      onError: (e) => _failAll(NslException('socket_error', '$e')),
      onDone: () => _failAll(NslException('disconnected')),
      cancelOnError: true,
    );

    await channel.ready.timeout(timeout);
    await _ready!.future.timeout(timeout);
  }

  void _onMessage(dynamic raw) {
    final Map<String, dynamic> msg;
    try {
      msg = jsonDecode(raw as String) as Map<String, dynamic>;
    } catch (_) {
      return; // binary or malformed — the server never sends either
    }
    switch (msg['type']) {
      case 'ready':
        if (_ready?.isCompleted == false) _ready!.complete();
      case 'ack':
        _ackController.add(NslAck.fromJson(msg));
      case 'result':
        _pending?.complete(NslResult.fromJson(msg));
        _pending = null;
      case 'reset_ok':
        _resetDone?.complete();
        _resetDone = null;
      case 'error':
        final err = NslException(
          msg['error'] as String? ?? 'error',
          msg['detail'] as String?,
        );
        // An error after `done` (too_short, server_busy) resolves that request;
        // otherwise it is unsolicited and there is nothing waiting on it.
        if (_pending != null) {
          _pending!.completeError(err);
          _pending = null;
        }
    }
  }

  void _failAll(Object error) {
    if (_ready?.isCompleted == false) _ready!.completeError(error);
    _pending?.completeError(error);
    _pending = null;
    _resetDone?.complete();
    _resetDone = null;
  }

  /// Fire-and-forget: acks arrive on [acks]. Deliberately not awaited — waiting
  /// for each ack would pace the camera at the network round-trip instead of
  /// the capture rate.
  void sendFrame(Uint8List jpeg) => _channel?.sink.add(jpeg);

  Future<NslResult> finish({Duration timeout = const Duration(seconds: 20)}) {
    if (_channel == null) return Future.error(NslException('not_connected'));
    _pending = Completer<NslResult>();
    _channel!.sink.add(jsonEncode({'type': 'done'}));
    return _pending!.future.timeout(timeout, onTimeout: () {
      _pending = null;
      throw NslException('timeout', 'no result within ${timeout.inSeconds}s');
    });
  }

  Future<void> reset({Duration timeout = const Duration(seconds: 5)}) {
    if (_channel == null) return Future.value();
    _resetDone = Completer<void>();
    _channel!.sink.add(jsonEncode({'type': 'reset'}));
    return _resetDone!.future.timeout(timeout, onTimeout: () {});
  }

  Future<void> dispose() async {
    await _sub?.cancel();
    _sub = null;
    await _channel?.sink.close();
    _channel = null;
    _pending = null;
    _ready = null;
    _resetDone = null;
  }
}
