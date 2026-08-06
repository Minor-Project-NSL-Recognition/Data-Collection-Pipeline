import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';

import 'frame_encoder.dart';
import 'nsl_client.dart';

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

  bool _initialising = true;
  String? _fatal;
  bool _connected = false;
  bool _recording = false;
  bool _finishing = false;

  // live telemetry
  bool _pose = false;
  bool _hands = false;
  int _kept = 0;
  int _sent = 0;
  DateTime? _startedAt;
  DateTime _lastFrameAt = DateTime.fromMillisecondsSinceEpoch(0);

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
      final front = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        front,
        ResolutionPreset.medium, // ~640x480; lower hurts hand landmarks
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await controller.initialize();
      final encoder = await FrameEncoder.spawn();
      if (!mounted) return;
      setState(() {
        _camera = controller;
        _encoder = encoder;
        _initialising = false;
      });
      await _connect();
    } catch (e) {
      if (mounted) setState(() => _fatal = '$e', );
    }
  }

  Future<void> _connect() async {
    setState(() {
      _connected = false;
      _error = null;
    });
    try {
      await _client.connect(widget.serverUrl, apiKey: widget.apiKey);
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
    if (!_connected || _camera == null || _recording) return;
    await _client.reset();
    setState(() {
      _recording = true;
      _result = null;
      _error = null;
      _pose = false;
      _hands = false;
      _kept = 0;
      _sent = 0;
      _startedAt = DateTime.now();
    });
    await _camera!.startImageStream(_onFrame);
  }

  void _onFrame(CameraImage image) {
    if (!_recording || _encoder == null) return;
    final now = DateTime.now();
    if (now.difference(_lastFrameAt) < _frameInterval) return;
    if (_encoder!.isBusy) return; // still encoding the previous frame
    _lastFrameAt = now;

    _encoder!
        .encode(image, rotation: _camera!.description.sensorOrientation)
        .then((jpeg) {
      if (jpeg == null || !_recording) return;
      _client.sendFrame(jpeg);
      _sent++;
    });
  }

  Future<void> _stop() async {
    if (!_recording) return;
    setState(() {
      _recording = false;
      _finishing = true;
    });
    try {
      await _camera?.stopImageStream();
    } catch (_) {/* already stopped */}

    try {
      final result = await _client.finish();
      if (!mounted) return;
      setState(() {
        _result = result;
        _finishing = false;
      });
      // Speak ONLY on `accepted`. The softmax is closed-world and names a
      // phrase for anything; `unknown` is the open-set gate rejecting it.
      if (result.isAccepted) {
        await _tts.setSpeechRate(0.45);
        await _tts.speak(result.display ?? result.bestGuess);
      }
    } on NslException catch (e) {
      if (mounted) {
        setState(() {
          _finishing = false;
          _error = e.toString();
        });
      }
      if (e.code == 'disconnected' || e.code == 'socket_error') _connect();
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
    final camera = _camera!;
    // Mirror the *preview only* so signing feels natural. The frames sent to
    // the server stay unmirrored — the server flips them itself, matching
    // record.py. Mirroring here as well would flip them back.
    final isFront = camera.description.lensDirection == CameraLensDirection.front;
    return ClipRect(
      child: Transform.scale(
        scaleX: isFront ? -1 : 1,
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
        body: 'Hold the button and perform one sign.',
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
    final enabled = _connected && !_finishing;
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 28),
      child: GestureDetector(
        onTapDown: enabled ? (_) => _start() : null,
        onTapUp: enabled ? (_) => _stop() : null,
        onTapCancel: enabled ? _stop : null,
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
              !_connected
                  ? 'No server'
                  : _recording
                      ? 'Signing…  release to recognise'
                      : 'Hold to sign',
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
