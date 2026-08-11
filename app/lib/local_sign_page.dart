import 'dart:async';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'landmarker.dart';
import 'local_recognizer.dart';
import 'rgba_converter.dart';
import 'widgets.dart';

/// Fully offline recognition. No network, no server, no internet.
///
/// The pipeline is the same one `server/` runs, moved onto the phone:
///
///   camera frame -> upright mirrored RGBA (isolate)
///                -> MediaPipe Pose + Hand landmarkers (native)  == 225-vector
///                -> normalize + standardize (Dart)
///                -> BiLSTM via TFLite (Dart)
///                -> Mahalanobis open-set gate -> decision -> speech
///
/// Landmarks are extracted *while the user signs*, exactly like the streaming
/// server did, so the only work left at "stop" is normalize + infer.
const _prefsUseFront = 'camera_use_front';
const _prefsMirrorPreview = 'camera_mirror_preview';

/// Frame pacing. The training clips were captured at a median 15.7 fps and the
/// model reads frame COUNT as duration, so a clip fed at 30 fps is a temporal
/// scale it never saw. 16 sits just above the target so jitter cannot starve it.
///
/// Unlike the server path there is no second decimator downstream — this
/// interval *is* the rate that reaches the model.
const _targetFps = 16;
const _frameInterval = Duration(milliseconds: 1000 ~/ _targetFps);

class LocalSignPage extends StatefulWidget {
  const LocalSignPage({super.key, required this.onUseServer});

  /// Switch to the networked server page. Kept as an escape hatch: if the phone
  /// turns out to be too slow, the laptop can still do the work.
  final VoidCallback onUseServer;

  @override
  State<LocalSignPage> createState() => _LocalSignPageState();
}

class _LocalSignPageState extends State<LocalSignPage>
    with WidgetsBindingObserver {
  CameraController? _camera;
  RgbaConverter? _converter;
  final _landmarker = Landmarker();
  LocalRecognizer? _recognizer;
  final _tts = FlutterTts();

  List<CameraDescription> _cameras = const [];
  int _cameraIndex = 0;

  bool _initialising = true;
  String? _fatal;
  bool _recording = false;
  bool _finishing = false;
  bool _switchingCamera = false;
  bool _mirrorPreview = false;
  bool _busy = false;

  /// The clip being accumulated: raw 225-vectors, in `record.py`'s form.
  final List<Float32List> _clip = [];

  // live telemetry
  bool _pose = false;
  bool _hands = false;
  int _seen = 0;

  /// Stage counters, so "frames 0" is diagnosable instead of just blank.
  /// `cam` counting up with `conv` at zero means the converter is failing;
  /// `conv` up with `frames` at zero means the landmarker is. Without these,
  /// every distinct failure looks identical from the outside.
  int _camFrames = 0;
  int _converted = 0;
  int _dropped = 0;

  /// First error from the frame pipeline, kept verbatim. Previously these were
  /// swallowed by a bare `catchError`, which made a systematic failure
  /// indistinguishable from an idle camera.
  String? _frameError;
  int _frameErrors = 0;

  /// Filename of the last raw clip written to external files, shown so it can be
  /// matched up with what `adb pull` retrieves.
  String? _savedClip;

  /// Dimensions of the upright frame the landmarks were extracted from. Needed to
  /// undo the aspect mismatch, since the correction depends on height/width.
  int? _frameW;
  int? _frameH;

  /// The two on-device corrections, individually switchable at runtime.
  ///
  /// Both were derived from measurements on dumped clips, and both changed
  /// behaviour in ways that were hard to attribute without being able to isolate
  /// them. Toggling on the device beats a rebuild per hypothesis: tap, sign, read
  /// the answer. Persisted so a chosen configuration survives a restart.
  bool _fixAspect = true;
  bool _fixResample = true;

  /// Whether the Mahalanobis distance may REJECT a sign. Off by default: measured
  /// on-device distances (20-27) sit past the largest training distance (20.8), and
  /// no threshold separates real signs from non-signs there. Left toggleable so the
  /// decision stays visible rather than buried.
  bool _enforceOod = false;

  static const _prefsFixAspect = 'fix_aspect';
  static const _prefsFixResample = 'fix_resample';
  static const _prefsEnforceOod = 'enforce_ood';
  DateTime? _startedAt;
  DateTime _lastFrameAt = DateTime.fromMillisecondsSinceEpoch(0);
  int _convertMs = 0;
  int _detectMs = 0;
  String? _sizeLabel;

  LocalResult? _result;
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
      _fixAspect = prefs.getBool(_prefsFixAspect) ?? true;
      _fixResample = prefs.getBool(_prefsFixResample) ?? true;
      _enforceOod = prefs.getBool(_prefsEnforceOod) ?? false;

      // Both are slow and independent: ~18 MB of model bundles between them.
      final converter = await RgbaConverter.spawn();
      final recognizer = await LocalRecognizer.load();
      await _landmarker.init();

      if (!mounted) {
        converter.dispose();
        recognizer.dispose();
        return;
      }
      _cameras = cameras;
      _converter = converter;
      _recognizer = recognizer;
      _mirrorPreview = mirror;
      await _openCamera(_indexForLens(
        wantFront ? CameraLensDirection.front : CameraLensDirection.back,
      ));
      if (!mounted) return;
      setState(() => _initialising = false);
    } catch (e) {
      if (mounted) setState(() => _fatal = '$e');
    }
  }

  int _indexForLens(CameraLensDirection lens) {
    final i = _cameras.indexWhere((c) => c.lensDirection == lens);
    return i < 0 ? 0 : i;
  }

  /// One camera per lens direction. Phones expose several back cameras (wide,
  /// ultrawide, macro) and cycling all of them makes the switch button feel
  /// broken.
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

  /// Flip one of the correction toggles. Blocked mid-recording so a clip is never
  /// captured under one configuration and scored under another.
  Future<void> _toggleFix(String which) async {
    if (_recording) return;
    late final String key;
    late final bool value;
    setState(() {
      switch (which) {
        case 'aspect':
          _fixAspect = !_fixAspect;
          key = _prefsFixAspect;
          value = _fixAspect;
        case 'resample':
          _fixResample = !_fixResample;
          key = _prefsFixResample;
          value = _fixResample;
        case 'ood':
          _enforceOod = !_enforceOod;
          key = _prefsEnforceOod;
          value = _enforceOod;
      }
    });
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(key, value);
  }

  Future<void> _toggleMirror() async {
    final next = !_mirrorPreview;
    setState(() => _mirrorPreview = next);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_prefsMirrorPreview, next);
  }

  /// Flip between front and back.
  ///
  /// The frames handed to MediaPipe stay mirrored for BOTH lenses. The lens faces
  /// the signer either way, so a raw frame has the signer's right hand on the
  /// image-left in both cases; `record.py` mirrored every training clip. Making
  /// the mirror depend on the lens would swap the left and right hand blocks of
  /// the 225-vector for one of them.
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

  Future<void> _start() async {
    if (_camera == null || _recording || _busy || _switchingCamera) return;
    if (_recognizer == null || !_landmarker.isReady) return;
    setState(() {
      _busy = true;
      _result = null;
      _error = null;
      _pose = false;
      _hands = false;
      _seen = 0;
      _camFrames = 0;
      _converted = 0;
      _dropped = 0;
      _frameError = null;
      _frameErrors = 0;
      _clip.clear();
    });
    try {
      await _landmarker.reset();
      await _camera!.startImageStream(_onFrame);
      if (!mounted) return;
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
    final converter = _converter;
    if (!_recording || converter == null || camera == null) return;

    _camFrames++;

    final now = DateTime.now();
    if (now.difference(_lastFrameAt) < _frameInterval) return;
    if (converter.isBusy) return; // still converting the previous frame
    _lastFrameAt = now;

    // Deliberately not awaited: this is a camera callback, which must return
    // immediately. Errors are handled inside rather than by a trailing
    // catchError, so the analyzer's FutureOr<Null> contract stays satisfied and
    // nothing can escape unhandled.
    unawaited(_processFrame(image, camera, converter));
  }

  Future<void> _processFrame(
    CameraImage image,
    CameraController camera,
    RgbaConverter converter,
  ) async {
    try {
      // Portrait is locked in main.dart, so sensorOrientation alone is the
      // correct upright rotation for both lenses.
      final frame = await converter.convert(
        image,
        rotation: camera.description.sensorOrientation,
      );
      if (frame == null || !_recording) return;
      _converted++;
      _convertMs = frame.millis;
      _frameW = frame.width;
      _frameH = frame.height;
      _sizeLabel = '${frame.srcWidth}x${frame.srcHeight}'
          '>${frame.width}x${frame.height}';

      final t0 = DateTime.now();
      final lm = await _landmarker.detect(
        rgba: frame.bytes,
        width: frame.width,
        height: frame.height,
        // Elapsed clip time, which is what the detectors' trackers want.
        timestampMs: _startedAt == null
            ? 0
            : DateTime.now().difference(_startedAt!).inMilliseconds,
      );
      if (lm == null) {
        _dropped++; // native backpressure: previous frame still in the detectors
        return;
      }
      if (!_recording) return;

      _clip.add(lm.vector);
      _seen++;
      if (!mounted) return;
      setState(() {
        _detectMs = DateTime.now().difference(t0).inMilliseconds;
        _pose = lm.pose;
        _hands = lm.anyHand;
      });
    } catch (e, s) {
      _onFrameError(e, s);
    }
  }

  /// Record the first frame-pipeline error and count the rest.
  ///
  /// This runs up to ~16x/second, so it must not spam setState or the log — but
  /// swallowing it outright (which is what this code did originally) turned a
  /// systematic failure into a silent "frames 0" with no way to tell whether the
  /// camera, the converter or the landmarker was at fault.
  void _onFrameError(Object error, StackTrace stack) {
    _frameErrors++;
    if (_frameError != null) return; // already reported; just keep counting
    _frameError = '$error';
    debugPrint('NSL frame pipeline failed: $error\n$stack');
    if (mounted) setState(() {});
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

      final recognizer = _recognizer;
      if (recognizer == null) throw StateError('recognizer not loaded');
      if (_clip.length < 5) {
        throw StateError(
            'only ${_clip.length} usable frames — hold the sign a little longer');
      }

      // The clip is normalized in place, so hand over a copy and clear ours.
      final clip = List<Float32List>.from(_clip);
      _clip.clear();

      // Dump the RAW clip before predict() normalizes it in place. Always on:
      // these are ~80 KB and they are the only way to compare what the phone
      // actually fed the model against what live_demo.py feeds it. record.py's
      // landmarks were inspectable; the phone's were not, which is precisely
      // what made on-device disagreement so hard to diagnose.
      final raw = clip.map((f) => Float32List.fromList(f)).toList();
      final stamp = DateTime.now()
          .toIso8601String()
          .replaceAll(':', '-')
          .replaceAll('.', '-');
      _landmarker.saveClip(raw, 'clip_$stamp.f32').then((path) {
        if (mounted) setState(() => _savedClip = path);
      }).catchError((Object e) {
        debugPrint('clip dump failed: $e');
        return null;
      });

      // Duration and frame size are what let predict() undo the two measured
      // on-device distortions: a capture rate the phone could not sustain, and
      // the portrait/landscape aspect mismatch.
      final durationSec = _startedAt == null
          ? null
          : DateTime.now().difference(_startedAt!).inMilliseconds / 1000.0;
      final result = recognizer.predict(
        clip,
        durationSec: _fixResample ? durationSec : null,
        frameWidth: _fixAspect ? _frameW : null,
        frameHeight: _fixAspect ? _frameH : null,
        enforceOodGate: _enforceOod,
      );
      if (!mounted) return;
      setState(() => _result = result);

      // Speak ONLY on `accepted`. Everything else is the model declining to
      // answer, and announcing a guess it rejected would be worse than silence.
      if (result.isAccepted) {
        await _tts.setSpeechRate(0.45);
        await _tts.speak(result.display ?? result.bestGuess);
      }
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    } finally {
      if (mounted) {
        setState(() {
          _finishing = false;
          _busy = false;
        });
      }
    }
  }

  double? get _fps {
    if (_startedAt == null || _seen == 0) return null;
    final secs = DateTime.now().difference(_startedAt!).inMilliseconds / 1000;
    return secs > 0.5 ? _seen / secs : null;
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive && _recording) _stop();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _camera?.dispose();
    _converter?.dispose();
    _recognizer?.dispose();
    _landmarker.close();
    _tts.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('NSL — Offline'),
        actions: [
          IconButton(
            tooltip: _mirrorPreview
                ? 'Preview: mirrored'
                : 'Preview: as others see you',
            icon: Icon(_mirrorPreview ? Icons.flip : Icons.flip_outlined),
            onPressed: _toggleMirror,
          ),
          if (_lensOptions.length > 1)
            IconButton(
              tooltip: _isFront
                  ? 'Switch to back camera'
                  : 'Switch to front camera',
              icon: const Icon(Icons.cameraswitch),
              onPressed:
                  (_recording || _switchingCamera) ? null : _switchCamera,
            ),
          IconButton(
            tooltip: 'Use the network server instead',
            icon: const Icon(Icons.cloud_outlined),
            onPressed: _recording ? null : widget.onUseServer,
          ),
        ],
      ),
      body: _fatal != null
          ? CenteredMessage(icon: Icons.error_outline, text: _fatal!)
          : _initialising
              ? const CenteredMessage(
                  icon: Icons.download_for_offline,
                  text: 'Loading on-device models…')
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
      return const CenteredMessage(
          icon: Icons.cameraswitch, text: 'Switching camera…');
    }
    // Preview orientation only. The frames fed to MediaPipe are mirrored
    // independently of this and are never affected by the toggle.
    return ClipRect(
      child: Transform.scale(
        scaleX: _mirrorPreview ? -1 : 1,
        child: Center(child: CameraPreview(camera)),
      ),
    );
  }

  Widget _telemetry() {
    final fps = _fps;
    return Container(
      color: Colors.grey.shade900,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      child: Wrap(
        spacing: 8,
        runSpacing: 4,
        children: [
          const StatusChip('offline', true),
          // "gate" now reports whether the distance may REJECT, not merely whether
          // stats were loaded — the loaded-but-unused state was misleading.
          StatusChip('gate', _enforceOod,
              onTap: _recording ? null : () => _toggleFix('ood')),
          // Tap either to turn that correction off, then repeat the sign. Four
          // combinations, so whichever one behaves is the answer.
          StatusChip('aspect', _fixAspect,
              onTap: _recording ? null : () => _toggleFix('aspect')),
          StatusChip('resample', _fixResample,
              onTap: _recording ? null : () => _toggleFix('resample')),
          StatusChip('pose', _pose),
          StatusChip('hands', _hands),
          InfoChip('frames', '$_seen'),
          // cam -> conv -> frames. Whichever number stops climbing is the stage
          // that is broken.
          InfoChip('cam', '$_camFrames'),
          InfoChip('conv', '$_converted'),
          if (_dropped > 0) InfoChip('busy-drop', '$_dropped'),
          if (_frameErrors > 0) StatusChip('err $_frameErrors', false),
          if (fps != null) InfoChip('fps', fps.toStringAsFixed(1)),
          // Where the ~62 ms budget goes. If detect dominates, the phone's
          // landmarkers are the limit and maxWidth is the lever.
          if (_convertMs > 0) InfoChip('conv', '${_convertMs}ms'),
          if (_detectMs > 0) InfoChip('detect', '${_detectMs}ms'),
          if (_sizeLabel != null) InfoChip('src', _sizeLabel!),
          InfoChip('lens', _isFront ? 'front' : 'back'),
        ],
      ),
    );
  }

  Widget _resultPanel() {
    if (_error != null) {
      return ResultBanner(
          color: Colors.red.shade900, title: 'Error', body: _error!);
    }
    // A frame-pipeline failure means no clip is being built at all, so show it
    // even while recording rather than waiting for a stop that cannot succeed.
    if (_frameError != null && _seen == 0) {
      return ResultBanner(
        color: Colors.red.shade900,
        title: 'Frame pipeline failed ($_frameErrors)',
        body: _frameError!,
      );
    }
    if (_finishing) {
      return const ResultBanner(
          color: Colors.blueGrey, title: 'Recognising…', body: '');
    }
    final r = _result;
    if (r == null) {
      return const ResultBanner(
        color: Colors.black54,
        title: 'Ready — no network needed',
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
    final dumped = _savedClip == null
        ? ''
        : '\nsaved ${_savedClip!.split('/').last}';
    return ResultBanner(
      color: color,
      title: title,
      body: '$top3\n${r.nFrames}'
          '${r.framesUsed != r.nFrames ? '->${r.framesUsed}' : ''}'
          ' frames · ${r.standardize} · ${r.inferenceMs}ms'
          '${r.oodDistance != null ? ' · distance ${r.oodDistance!.toStringAsFixed(1)}' : ''}'
          '$dumped',
    );
  }

  Widget _controls() {
    final ready = !_initialising && _recognizer != null && _landmarker.isReady;
    final enabled = ready && !_finishing && !_busy;
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
              !ready
                  ? 'Loading…'
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
