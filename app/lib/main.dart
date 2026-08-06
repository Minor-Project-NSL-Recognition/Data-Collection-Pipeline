import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'sign_page.dart';

const _prefsKey = 'server_url';
const _prefsApiKey = 'api_key';
const _defaultUrl = 'http://192.168.1.100:8000';

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
        _url = p.getString(_prefsKey);
        _apiKey = p.getString(_prefsApiKey);
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

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NSL Recognition',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true),
      home: _loading
          ? const Scaffold(body: Center(child: CircularProgressIndicator()))
          : (_url == null || _editing)
              ? ServerSetupPage(
                  // Prefill with what is already stored, so reopening settings
                  // after a tunnel/IP change is an edit, not a re-entry.
                  initialUrl: _url ?? _defaultUrl,
                  initialApiKey: _apiKey ?? '',
                  onSaved: _save,
                  onCancel:
                      _url == null ? null : () => setState(() => _editing = false),
                )
              : SignPage(
                  serverUrl: _url!,
                  apiKey: _apiKey,
                  onEditServer: () => setState(() => _editing = true),
                ),
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
  });
  final String initialUrl;
  final String initialApiKey;
  final void Function(String url, String apiKey) onSaved;

  /// Null on first launch, when there is nothing to go back to.
  final VoidCallback? onCancel;

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
              'need to be on the same network at all.',
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
              ),
              onSubmitted: (_) => _submit(),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _keyController,
              autocorrect: false,
              decoration: const InputDecoration(
                labelText: 'API key (optional)',
                helperText: 'Required only if the server sets NSL_API_KEY',
                border: OutlineInputBorder(),
              ),
              onSubmitted: (_) => _submit(),
            ),
            const SizedBox(height: 16),
            FilledButton(onPressed: _submit, child: const Text('Connect')),
            if (widget.onCancel != null)
              TextButton(onPressed: widget.onCancel, child: const Text('Cancel')),
          ],
        ),
      ),
    );
  }
}
