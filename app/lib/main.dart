import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'local_sign_page.dart';
import 'sign_page.dart';

const _prefsKey = 'server_url';
const _prefsApiKey = 'api_key';

/// `offline` (everything on the phone) or `server` (stream to the Python
/// service). Defaults to offline: the app is for emergencies, where assuming a
/// network is the wrong default. Server mode stays available as an escape hatch
/// for a phone too slow to run the landmarkers.
const _prefsMode = 'inference_mode';
const _modeOffline = 'offline';
const _modeServer = 'server';

/// Baked in at build time, so a freshly installed APK connects with no setup:
///
///     flutter build apk --release \
///       --dart-define=NSL_SERVER_URL=https://nsl.example.com \
///       --dart-define=NSL_API_KEY=...
///
/// Use with a *named* tunnel, whose hostname survives restarts. A quick tunnel's
/// URL is regenerated every time, so baking one in would go stale immediately.
/// Empty (the default) falls back to the manual setup screen.
const _bakedUrl = String.fromEnvironment('NSL_SERVER_URL');
const _bakedApiKey = String.fromEnvironment('NSL_API_KEY');

/// Only ever a placeholder in the setup field — never used to connect.
const _placeholderUrl = 'http://192.168.1.100:8000';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // The model reads frame count as duration and was trained on upright,
  // front-facing clips; letting the device rotate mid-sign would change the
  // frame geometry underneath the recogniser.
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(const NslApp());
}

class NslApp extends StatefulWidget {
  const NslApp({super.key});
  @override
  State<NslApp> createState() => _NslAppState();
}

class _NslAppState extends State<NslApp> {
  String? _url;
  String? _apiKey;
  String _mode = _modeOffline;
  bool _loading = true;
  // Distinct from `_url == null`. Opening the settings must not throw away the
  // saved URL — otherwise reopening them means retyping it from scratch.
  bool _editing = false;

  @override
  void initState() {
    super.initState();
    SharedPreferences.getInstance().then((p) {
      if (!mounted) return;
      setState(() {
        // A URL saved from the gear icon wins: it is a deliberate override, and
        // silently replacing it on the next build would be baffling. Reinstall
        // (or Clear storage) to fall back to the built-in one.
        _url = p.getString(_prefsKey) ?? (_bakedUrl.isEmpty ? null : _bakedUrl);
        _apiKey =
            p.getString(_prefsApiKey) ?? (_bakedApiKey.isEmpty ? null : _bakedApiKey);
        _mode = p.getString(_prefsMode) ?? _modeOffline;
        _loading = false;
      });
    });
  }

  Future<void> _save(String url, String apiKey) async {
    final p = await SharedPreferences.getInstance();
    await p.setString(_prefsKey, url);
    await p.setString(_prefsApiKey, apiKey);
    if (mounted) {
      setState(() {
        _url = url;
        _apiKey = apiKey;
        _editing = false;
      });
    }
  }

  Future<void> _setMode(String mode) async {
    final p = await SharedPreferences.getInstance();
    await p.setString(_prefsMode, mode);
    if (mounted) setState(() => _mode = mode);
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NSL Recognition',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true),
      home: _loading
          ? const Scaffold(body: Center(child: CircularProgressIndicator()))
          : _buildHome(),
    );
  }

  Widget _buildHome() {
    if (_mode == _modeOffline) {
      return LocalSignPage(onUseServer: () => _setMode(_modeServer));
    }
    // Server mode needs somewhere to connect before it can do anything.
    if (_url == null || _editing) {
      return ServerSetupPage(
        // Prefill with what is already stored, so reopening settings after a
        // tunnel/IP change is an edit, not a re-entry.
        initialUrl: _url ?? _placeholderUrl,
        initialApiKey: _apiKey ?? '',
        onSaved: _save,
        onCancel: _url == null
            // Nothing to go back to, so offer the way out that always works.
            ? () => _setMode(_modeOffline)
            : () => setState(() => _editing = false),
        cancelLabel: _url == null ? 'Use offline mode' : 'Cancel',
      );
    }
    return SignPage(
      serverUrl: _url!,
      apiKey: _apiKey,
      onEditServer: () => setState(() => _editing = true),
      onUseOffline: () => _setMode(_modeOffline),
    );
  }
}

class ServerSetupPage extends StatefulWidget {
  const ServerSetupPage({
    super.key,
    required this.initialUrl,
    required this.initialApiKey,
    required this.onSaved,
    this.onCancel,
    this.cancelLabel = 'Cancel',
  });
  final String initialUrl;
  final String initialApiKey;
  final void Function(String url, String apiKey) onSaved;

  /// Null only if there is genuinely nowhere to go. On first launch this returns
  /// to offline mode rather than being absent, so the user is never stranded on
  /// a form demanding a server they may not have.
  final VoidCallback? onCancel;

  /// What that button says — "Cancel" when editing, "Use offline mode" when this
  /// is the first launch and offline is the real alternative.
  final String cancelLabel;

  @override
  State<ServerSetupPage> createState() => _ServerSetupPageState();
}

class _ServerSetupPageState extends State<ServerSetupPage> {
  late final _controller = TextEditingController(text: widget.initialUrl);
  late final _keyController = TextEditingController(text: widget.initialApiKey);
  String? _error;

  void _submit() {
    final text = _controller.text.trim();
    final uri = Uri.tryParse(text);
    if (uri == null || !uri.hasScheme || uri.host.isEmpty) {
      setState(() => _error = 'Expected something like http://192.168.1.7:8000');
      return;
    }
    widget.onSaved(
      text.replaceAll(RegExp(r'/+$'), ''),
      _keyController.text.trim(),
    );
  }

  /// A quick tunnel's hostname is 40-odd random characters that changes on every
  /// restart. Typing that on a phone is the worst part of the whole workflow, so
  /// send it to the phone however you like and paste it here.
  Future<void> _paste(TextEditingController target) async {
    final data = await Clipboard.getData(Clipboard.kTextPlain);
    final text = data?.text?.trim();
    if (text == null || text.isEmpty) return;
    target.text = text;
    if (mounted) setState(() => _error = null);
  }

  /// Back to whatever was compiled in with --dart-define, without a reinstall.
  void _useBuiltIn() {
    _controller.text = _bakedUrl;
    _keyController.text = _bakedApiKey;
    setState(() => _error = null);
  }

  @override
  void dispose() {
    _controller.dispose();
    _keyController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Server')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text('Where is the recognition server?'),
            const SizedBox(height: 8),
            const Text(
              'Same Wi-Fi:  http://<laptop-ip>:8000 — find it with `ipconfig` '
              '(Windows) or `hostname -I` (Linux). Not 127.0.0.1; on the phone '
              'that would mean the phone itself.\n\n'
              'Cloudflare tunnel:  the https://… URL. Then the phone does not '
              'need to be on the same network at all. A quick tunnel gets a new '
              'URL every restart — send it to the phone and use the paste '
              'button rather than typing it.',
              style: TextStyle(fontSize: 12, color: Colors.white54),
            ),
            const SizedBox(height: 20),
            TextField(
              controller: _controller,
              keyboardType: TextInputType.url,
              autocorrect: false,
              decoration: InputDecoration(
                labelText: 'Server URL',
                border: const OutlineInputBorder(),
                errorText: _error,
                suffixIcon: IconButton(
                  tooltip: 'Paste',
                  icon: const Icon(Icons.content_paste),
                  onPressed: () => _paste(_controller),
                ),
              ),
              onSubmitted: (_) => _submit(),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _keyController,
              autocorrect: false,
              decoration: InputDecoration(
                labelText: 'API key (optional)',
                helperText: 'Required only if the server sets NSL_API_KEY',
                border: const OutlineInputBorder(),
                suffixIcon: IconButton(
                  tooltip: 'Paste',
                  icon: const Icon(Icons.content_paste),
                  onPressed: () => _paste(_keyController),
                ),
              ),
              onSubmitted: (_) => _submit(),
            ),
            const SizedBox(height: 16),
            FilledButton(onPressed: _submit, child: const Text('Connect')),
            if (_bakedUrl.isNotEmpty)
              TextButton(
                onPressed: _useBuiltIn,
                child: const Text('Use built-in server'),
              ),
            if (widget.onCancel != null)
              TextButton(
                onPressed: widget.onCancel,
                child: Text(widget.cancelLabel),
              ),
          ],
        ),
      ),
    );
  }
}
