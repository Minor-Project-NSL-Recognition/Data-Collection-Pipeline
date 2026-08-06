import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'sign_page.dart';

const _prefsKey = 'server_url';
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
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    SharedPreferences.getInstance().then((p) {
      if (!mounted) return;
      setState(() {
        _url = p.getString(_prefsKey);
        _loading = false;
      });
    });
  }

  Future<void> _save(String url) async {
    final p = await SharedPreferences.getInstance();
    await p.setString(_prefsKey, url);
    if (mounted) setState(() => _url = url);
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NSL Recognition',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true),
      home: _loading
          ? const Scaffold(body: Center(child: CircularProgressIndicator()))
          : _url == null
              ? ServerSetupPage(initial: _defaultUrl, onSaved: _save)
              : SignPage(
                  serverUrl: _url!,
                  onEditServer: () => setState(() => _url = null),
                ),
    );
  }
}

class ServerSetupPage extends StatefulWidget {
  const ServerSetupPage({super.key, required this.initial, required this.onSaved});
  final String initial;
  final ValueChanged<String> onSaved;

  @override
  State<ServerSetupPage> createState() => _ServerSetupPageState();
}

class _ServerSetupPageState extends State<ServerSetupPage> {
  late final _controller = TextEditingController(text: widget.initial);
  String? _error;

  void _submit() {
    final text = _controller.text.trim();
    final uri = Uri.tryParse(text);
    if (uri == null || !uri.hasScheme || uri.host.isEmpty) {
      setState(() => _error = 'Expected something like http://192.168.1.7:8000');
      return;
    }
    widget.onSaved(text.replaceAll(RegExp(r'/+$'), ''));
  }

  @override
  void dispose() {
    _controller.dispose();
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
            const Text(
              'Enter the address of the recognition server. The phone and the '
              'server must be on the same Wi-Fi network.',
            ),
            const SizedBox(height: 8),
            const Text(
              'Find it with `ipconfig` (Windows) or `hostname -I` (Linux) on the '
              'machine running the server. Not 127.0.0.1 — on the phone that '
              'would mean the phone itself.',
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
            const SizedBox(height: 16),
            FilledButton(onPressed: _submit, child: const Text('Connect')),
          ],
        ),
      ),
    );
  }
}
