import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'frame_encoder.dart';
import 'nsl_client.dart';

/// Which lens the user last chose. Front is the default because the signer is
/// usually alone; back is for when someone else holds the phone.
const _prefsUseFront = 'camera_use_front';

/// Whether to flip the preview horizontally. Default off: whether a front
/// preview arrives already mirrored is a platform/device decision, so the only
/// reliable answer is to let the signer pick the one that reads correctly and
/// remember it. Display only — never affects the frames sent to the server.
const _prefsMirrorPreview = 'camera_mirror_preview';

/// Aim slightly above the server's 15.7 fps decimation target so rounding and
/// jitter don't starve it. Sending faster is harmless (the server drops the
/// excess); sending much slower is not, because frame count encodes duration
/// and the training clips only ever ran 10.8-21.2 fps.
const _targetFps = 16;
const _frameInterval = Duration(milliseconds: 1000 ~/ _targetFps);

class SignPage extends StatefulWidget {
  const SignPage({
    super.key,
    required this.serverUrl,
    required this.onEditServer,
    this.apiKey,
  });
  final String serverUrl;
  final String? apiKey;
  final VoidCallback onEditServer;

  @override
  State<SignPage> createState() => _SignPageState();
}

class _SignPageState extends State<SignPage> with WidgetsBindingObserver {
  CameraController? _camera;
  FrameEncoder? _encoder;
  final _client = NslClient();
  final _tts = FlutterTts();
  StreamSubscription<NslAck>? _ackSub;
  StreamSubscription<NslException>? _closedSub;

  List<CameraDescription> _cameras = const [];
  int _cameraIndex = 0;

  bool _initialising = true;
  String? _fatal;
  bool _connected = false;
  bool _recording = false;
  bool _finishing = false;
  bool _switchingCamera = false;
  bool _mirrorPreview = false;

  /// True while [_start] or [_stop] is mid-flight.
  ///
  /// The button is a toggle, and both handlers await before the state that
  /// would stop a second tap is visible. Without this, a tap landing inside
  /// that window re-enters against half-built state: two `reset()` calls (the
  /// second orphans the first's completer, which then hangs its full timeout)
  /// and two `startImageStream()` calls (the second throws).
  bool _busy = false;

  // live telemetry
  bool _pose = false;
  bool _hands = false;
  int _kept = 0;
  int _sent = 0;
  DateTime? _startedAt;
  DateTime _lastFrameAt = DateTime.fromMillisecondsSinceEpoch(0);

  // Where the frame budget actually goes. Without these a slow client is
  // indistinguishable from a slow network or a slow server.
  int _encodeMs = 0;
  String? _sizeLabel;

  NslResult? _result;
  String? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _boot();
  }

  Future<void> _boot() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) throw Exception('no cameras on this device');
      final prefs = await SharedPreferences.getInstance();
      final wantFront = prefs.getBool(_prefsUseFront) ?? true;
      final mirror = prefs.getBool(_prefsMirrorPreview) ?? false;

      final encoder = await FrameEncoder.spawn();
      if (!mounted) return;
      _cameras = cameras;
      _encoder = encoder;
      _mirrorPreview = mirror;
      await _openCamera(_indexForLens(
        wantFront ? CameraLensDirection.front : CameraLensDirection.back,
      ));
      if (!mounted) return;
      setState(() => _initialising = false);
      await _connect();
    } catch (e) {
      if (mounted) setState(() => _fatal = '$e', );
    }
  }

  /// First camera with [lens], falling back to whatever the device has.
  int _indexForLens(CameraLensDirection lens) {
    final i = _cameras.indexWhere((c) => c.lensDirection == lens);
    return i < 0 ? 0 : i;
  }

  /// One camera per lens direction, in a stable order. Phones commonly expose
  /// several back cameras (wide, ultrawide, macro); cycling through all of them
  /// would make the switch button feel broken, so only the first of each counts.
  List<int> get _lensOptions {
    final seen = <CameraLensDirection>{};
    final out = <int>[];
    for (var i = 0; i < _cameras.length; i++) {
      if (seen.add(_cameras[i].lensDirection)) out.add(i);
    }
    return out;
  }

  bool get _isFront =>
      _camera?.description.lensDirection == CameraLensDirection.front;

  /// Tears down the current controller and brings up [index]. The encoder
  /// isolate is deliberately *not* recreated — it is stateless between frames,
  /// and respawning it would add a needless pause to every switch.
  Future<void> _openCamera(int index) async {
    final old = _camera;
    if (mounted) setState(() => _camera = null);
    if (old != null) {
      try {
        if (old.value.isStreamingImages) await old.stopImageStream();
      } catch (_) {/* already stopped */}
      await old.dispose();
    }

    final controller = CameraController(
      _cameras[index],
      ResolutionPreset.medium, // ~640x480; lower hurts hand landmarks
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    await controller.initialize();
    if (!mounted) {
      await controller.dispose();
      return;
    }
    setState(() {
      _camera = controller;
      _cameraIndex = index;
    });
  }

  Future<void> _toggleMirror() async {
    final next = !_mirrorPreview;
    setState(() => _mirrorPreview = next);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_prefsMirrorPreview, next);
  }

  /// Flip between front and back.
  ///
  /// The `mirror` flag sent to the server stays true for BOTH lenses, so no
  /// reconnect is needed. That is not an oversight: the lens faces the signer
  /// either way, so a raw frame has the signer's right hand on the image-left
  /// in both cases. `record.py` mirrored every training clip, and the server
  /// reproduces that flip. Making the flag depend on the lens would swap the
  /// left and right hand blocks in the 225-vector for one of them.
  Future<void> _switchCamera() async {
    if (_switchingCamera || _recording) return;
    final options = _lensOptions;
    if (options.length < 2) return;
    final pos = options.indexOf(_cameraIndex);
    final next = options[(pos + 1) % options.length];

    setState(() => _switchingCamera = true);
    try {
      await _openCamera(next);
      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool(_prefsUseFront,
          _cameras[next].lensDirection == CameraLensDirection.front);
    } catch (e) {
      if (mounted) setState(() => _error = 'Could not switch camera: $e');
    } finally {
      if (mounted) setState(() => _switchingCamera = false);
    }
  }

  Future<void> _connect() async {
    setState(() {
      _connected = false;
      _error = null;
    });
    try {
      await _client.connect(widget.serverUrl, apiKey: widget.apiKey);
      _closedSub?.cancel();
      _closedSub = _client.disconnects.listen((e) {
        if (!mounted) return;
        // Surface it the moment it happens. Tapping the `server` chip
        // reconnects, and _stop() also reconnects if a request fails.
        setState(() => _connected = false);
      });
      _ackSub?.cancel();
      _ackSub = _client.acks.listen((ack) {
        if (!mounted) return;
        setState(() {
          _kept = ack.nFrames;
          if (ack.accepted) {
            _pose = ack.pose;
            _hands = ack.hands;
          }
        });
      });
      if (mounted) setState(() => _connected = true);
    } catch (e) {
      if (mounted) {
        final unauthorized = '$e'.contains('unauthorized');
        setState(() {
          _connected = false;
          _error = unauthorized
              ? 'Server rejected the API key. Check it under the gear icon.'
              : 'Cannot reach ${widget.serverUrl}\n$e';
        });
      }
    }
  }

  Future<void> _start() async {
    if (!_connected || _camera == null || _recording || _busy || _switchingCamera) {
      return;
    }
    setState(() {
      _busy = true;
      _result = null;
      _error = null;
      _pose = false;
      _hands = false;
      _kept = 0;
      _sent = 0;
    });
    try {
      await _client.reset();
      await _camera!.startImageStream(_onFrame);
      if (!mounted) return;
      // Only now is the clip really running, so this is where the fps clock
      // starts and where _onFrame begins forwarding.
      setState(() {
        _recording = true;
        _startedAt = DateTime.now();
      });
    } catch (e) {
      if (mounted) setState(() => _error = 'Could not start: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  void _onFrame(CameraImage image) {
    final camera = _camera;
    if (!_recording || _encoder == null || camera == null) return;
    final now = DateTime.now();
    if (now.difference(_lastFrameAt) < _frameInterval) return;
    if (_encoder!.isBusy) return; // still encoding the previous frame
    _lastFrameAt = now;

    // Portrait is locked in main.dart, so sensorOrientation alone is the
    // correct upright rotation for both lenses.
    _encoder!
        .encode(image, rotation: camera.description.sensorOrientation)
        .then((frame) {
      if (frame == null || !_recording) return;
      _client.sendFrame(frame.bytes);
      _sent++;
      _encodeMs = frame.millis;
      _sizeLabel = '${frame.srcWidth}x${frame.srcHeight}'
          '>${frame.outWidth}x${frame.outHeight}';
      // A failed encode must not become an unhandled async error: this runs
      // ~16x/second, so one bad frame would otherwise flood the log mid-sign.
    }).catchError((_) {});
  }

  Future<void> _stop() async {
    if (!_recording || _busy) return;
    setState(() {
      _recording = false;
      _busy = true;
      _finishing = true;
    });
    try {
      try {
        await _camera?.stopImageStream();
      } catch (_) {/* already stopped */}

      final result = await _client.finish();
      if (!mounted) return;
      setState(() => _result = result);
      // Speak ONLY on `accepted`. The softmax is closed-world and names a
      // phrase for anything; `unknown` is the open-set gate rejecting it.
      if (result.isAccepted) {
        await _tts.setSpeechRate(0.45);
        await _tts.speak(result.display ?? result.bestGuess);
      }
    } on NslException catch (e) {
      if (mounted) setState(() => _error = e.toString());
      // A timeout means the socket is wedged even though it looks open, so
      // rebuild it rather than leaving the next sign to fail the same way.
      if (e.code == 'disconnected' ||
          e.code == 'socket_error' ||
          e.code == 'timeout') {
        _connect();
      }
    } catch (e) {
      // Anything else (a closed sink throwing StateError, a TTS failure) must
      // still land in `finally`, or the button stays disabled for good.
      if (mounted) setState(() => _error = 'Recognition failed: $e');
    } finally {
      if (mounted) {
        setState(() {
          _finishing = false;
          _busy = false;
        });
      }
    }
  }

  double? get _clientFps {
    if (_startedAt == null || _sent == 0) return null;
    final secs = DateTime.now().difference(_startedAt!).inMilliseconds / 1000;
    return secs > 0.5 ? _sent / secs : null;
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive && _recording) _stop();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _ackSub?.cancel();
    _closedSub?.cancel();
    _client.dispose();
    _camera?.dispose();
    _encoder?.dispose();
    _tts.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('NSL Recognition'),
        actions: [
          IconButton(
            tooltip: _mirrorPreview ? 'Preview: mirrored' : 'Preview: as others see you',
            icon: Icon(_mirrorPreview ? Icons.flip : Icons.flip_outlined),
            onPressed: _toggleMirror,
          ),
          if (_lensOptions.length > 1)
            IconButton(
              tooltip: _isFront ? 'Switch to back camera' : 'Switch to front camera',
              icon: const Icon(Icons.cameraswitch),
              // Blocked mid-sign: swapping the lens would restart the image
              // stream underneath a clip the server is still accumulating.
              onPressed: (_recording || _switchingCamera) ? null : _switchCamera,
            ),
          IconButton(
            tooltip: 'Server settings',
            icon: const Icon(Icons.settings),
            onPressed: widget.onEditServer,
          ),
        ],
      ),
      body: _fatal != null
          ? _CenteredMessage(icon: Icons.error_outline, text: _fatal!)
          : _initialising
              ? const _CenteredMessage(icon: Icons.hourglass_empty, text: 'Starting camera…')
              : Column(
                  children: [
                    Expanded(child: _preview()),
                    _telemetry(),
                    _resultPanel(),
                    _controls(),
                  ],
                ),
    );
  }

  Widget _preview() {
    final camera = _camera;
    if (camera == null || !camera.value.isInitialized) {
      return const _CenteredMessage(
          icon: Icons.cameraswitch, text: 'Switching camera…');
    }
    // Preview orientation only, toggled from the app bar and remembered.
    //
    // The frames sent to the server are never affected: they go unmirrored on
    // both lenses and the server flips them itself, matching record.py.
    // Mirroring the wire frames would flip them back and swap the left/right
    // hand blocks in the 225-vector.
    return ClipRect(
      child: Transform.scale(
        scaleX: _mirrorPreview ? -1 : 1,
        child: Center(child: CameraPreview(camera)),
      ),
    );
  }

  Widget _telemetry() {
    final fps = _clientFps;
    return Container(
      color: Colors.grey.shade900,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      child: Wrap(
        spacing: 8,
        runSpacing: 4,
        children: [
          _Chip('server', _connected, onTap: _connected ? null : _connect),
          _Chip('pose', _pose),
          _Chip('hands', _hands),
          _Info('frames', '$_kept'),
          if (fps != null) _Info('fps', fps.toStringAsFixed(1)),
          // enc is the whole client-side cost of one frame. If it is near the
          // 62 ms budget, the phone is the limit and maxWidth is the lever.
          if (_encodeMs > 0) _Info('enc', '${_encodeMs}ms'),
          if (_sizeLabel != null) _Info('src', _sizeLabel!),
          _Info('lens', _isFront ? 'front' : 'back'),
        ],
      ),
    );
  }

  Widget _resultPanel() {
    if (_error != null) {
      return _Banner(color: Colors.red.shade900, title: 'Error', body: _error!);
    }
    if (_finishing) {
      return const _Banner(color: Colors.blueGrey, title: 'Recognising…', body: '');
    }
    final r = _result;
    if (r == null) {
      return const _Banner(
        color: Colors.black54,
        title: 'Ready',
        body: 'Tap the button, perform one sign, then tap again.',
      );
    }
    final (color, title) = switch (r.status) {
      'accepted' => (Colors.green.shade800, r.display ?? r.bestGuess),
      'unknown' => (Colors.deepOrange.shade900, 'Unknown sign'),
      _ => (Colors.amber.shade900, 'Not confident'),
    };
    final top3 = r.top3
        .map((g) => '${g.label} ${(g.confidence * 100).round()}%')
        .join('   ');
    return _Banner(
      color: color,
      title: title,
      body: '$top3\n${r.nFrames} frames'
          '${r.oodDistance != null ? '   distance ${r.oodDistance!.toStringAsFixed(1)}' : ''}',
    );
  }

  Widget _controls() {
    final enabled = _connected && !_finishing && !_busy;
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 28),
      child: GestureDetector(
        // Tap to start, tap again to stop — not hold-to-record. Signing needs
        // both hands, so the signer cannot keep a finger on the button.
        onTap: enabled ? (_recording ? _stop : _start) : null,
        child: Container(
          height: 72,
          decoration: BoxDecoration(
            color: !enabled
                ? Colors.grey.shade800
                : _recording
                    ? Colors.red.shade700
                    : Colors.blue.shade700,
            borderRadius: BorderRadius.circular(36),
          ),
          child: Center(
            child: Text(
              // The transient states are named rather than left as a greyed
              // "Tap to sign", which reads as the app having ignored the tap.
              !_connected
                  ? 'No server'
                  : _finishing
                      ? 'Recognising…'
                      : _recording
                          ? 'Signing…  tap to recognise'
                          : _busy
                              ? 'Starting…'
                              : 'Tap to sign',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _Chip extends StatelessWidget {
  const _Chip(this.label, this.on, {this.onTap});
  final String label;
  final bool on;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: on ? Colors.green.shade700 : Colors.red.shade900,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(label, style: const TextStyle(color: Colors.white, fontSize: 12)),
      ),
    );
  }
}

class _Info extends StatelessWidget {
  const _Info(this.label, this.value);
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.grey.shade800,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text('$label $value',
          style: const TextStyle(color: Colors.white70, fontSize: 12)),
    );
  }
}

class _Banner extends StatelessWidget {
  const _Banner({required this.color, required this.title, required this.body});
  final Color color;
  final String title;
  final String body;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      color: color,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title,
              style: const TextStyle(
                  color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold)),
          if (body.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(body,
                  style: const TextStyle(color: Colors.white70, fontSize: 12)),
            ),
        ],
      ),
    );
  }
}

class _CenteredMessage extends StatelessWidget {
  const _CenteredMessage({required this.icon, required this.text});
  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: Colors.white54, size: 48),
            const SizedBox(height: 12),
            Text(text,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white70)),
          ],
        ),
      ),
    );
  }
}
