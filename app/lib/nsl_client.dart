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

/// Cloudflare (and most reverse proxies) close WebSockets that sit idle, and
/// between signs this one does exactly that. A periodic ping keeps it open;
/// without it the first sign works and the next one fails with a disconnect
/// some minutes later — an intermittent failure that is miserable to diagnose
/// live. Harmless on a LAN.
const _keepAlivePeriod = Duration(seconds: 30);

class NslClient {
  WebSocketChannel? _channel;
  StreamSubscription? _sub;
  Timer? _keepAlive;
  final _ackController = StreamController<NslAck>.broadcast();
  final _closedController = StreamController<NslException>.broadcast();
  Completer<NslResult>? _pending;
  Completer<void>? _ready;
  Completer<void>? _resetDone;

  Stream<NslAck> get acks => _ackController.stream;

  /// Fires when the socket drops. Without this the UI only finds out by
  /// timing out a request, so a connection that died while idle costs the
  /// user a 5 s reset plus a 20 s finish before anything is shown.
  Stream<NslException> get disconnects => _closedController.stream;

  bool get isConnected => _channel != null;

  /// [baseUrl] is the HTTP origin — `http://192.168.1.7:8000` on a LAN, or the
  /// `https://…` tunnel URL. `https` becomes `wss` automatically.
  ///
  /// [apiKey] is required when the server runs with `NSL_API_KEY` set, which it
  /// should whenever it is exposed through a tunnel. It goes in the query
  /// string rather than a header because WebSocket clients cannot reliably set
  /// headers on the upgrade request.
  ///
  /// [mirror] must stay true for a front camera: the server flips the frame to
  /// match `record.py`, which mirrored every training clip. Flipping twice puts
  /// the left and right hand blocks in each other's slots.
  Future<void> connect(
    String baseUrl, {
    bool mirror = true,
    String? apiKey,
    Duration timeout = const Duration(seconds: 15),
  }) async {
    await dispose();
    final ws = baseUrl.replaceFirst(RegExp(r'^http'), 'ws');
    final uri = Uri.parse('$ws/ws/stream').replace(queryParameters: {
      'mirror': mirror ? '1' : '0',
      if (apiKey != null && apiKey.isNotEmpty) 'key': apiKey,
    });

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

    _keepAlive = Timer.periodic(_keepAlivePeriod, (_) {
      if (_channel != null && _pending == null) {
        _channel!.sink.add(jsonEncode({'type': 'ping'}));
      }
    });
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

  void _failAll(NslException error) {
    _keepAlive?.cancel();
    _keepAlive = null;
    if (_ready?.isCompleted == false) _ready!.completeError(error);
    if (_pending?.isCompleted == false) _pending!.completeError(error);
    _pending = null;
    if (_resetDone?.isCompleted == false) _resetDone!.complete();
    _resetDone = null;
    if (_closedController.hasListener) _closedController.add(error);
  }

  /// Fire-and-forget: acks arrive on [acks]. Deliberately not awaited — waiting
  /// for each ack would pace the camera at the network round-trip instead of
  /// the capture rate.
  ///
  /// A socket that dies mid-clip would otherwise throw once per frame, ~16
  /// times a second, from a callback with nowhere to report. Dropping the frame
  /// is right: [finish] still reports the real failure, and [disconnects] has
  /// already fired.
  void sendFrame(Uint8List jpeg) {
    try {
      _channel?.sink.add(jpeg);
    } catch (_) {/* connection is going away; finish() will surface it */}
  }

  Future<NslResult> finish({Duration timeout = const Duration(seconds: 20)}) {
    final channel = _channel;
    if (channel == null) return Future.error(NslException('not_connected'));
    // Never drop a live completer on the floor: whoever is awaiting it would
    // wait out its full timeout for a reply that is now addressed to us.
    if (_pending?.isCompleted == false) {
      _pending!.completeError(NslException('superseded'));
    }
    final pending = Completer<NslResult>();
    _pending = pending;
    try {
      channel.sink.add(jsonEncode({'type': 'done'}));
    } catch (e) {
      // Adding to a closed sink throws synchronously, which would otherwise
      // escape as an unhandled error rather than a failed request.
      _pending = null;
      return Future.error(NslException('socket_error', '$e'));
    }
    return pending.future.timeout(timeout, onTimeout: () {
      if (identical(_pending, pending)) _pending = null;
      throw NslException('timeout', 'no result within ${timeout.inSeconds}s');
    });
  }

  Future<void> reset({Duration timeout = const Duration(seconds: 5)}) {
    final channel = _channel;
    if (channel == null) return Future.value();
    if (_resetDone?.isCompleted == false) _resetDone!.complete();
    final done = Completer<void>();
    _resetDone = done;
    try {
      channel.sink.add(jsonEncode({'type': 'reset'}));
    } catch (e) {
      _resetDone = null;
      return Future.error(NslException('socket_error', '$e'));
    }
    return done.future.timeout(timeout, onTimeout: () {
      if (identical(_resetDone, done)) _resetDone = null;
    });
  }

  Future<void> dispose() async {
    _keepAlive?.cancel();
    _keepAlive = null;
    await _sub?.cancel();
    _sub = null;
    await _channel?.sink.close();
    _channel = null;
    _pending = null;
    _ready = null;
    _resetDone = null;
  }
}
