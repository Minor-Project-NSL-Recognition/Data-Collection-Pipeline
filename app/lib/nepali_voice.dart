import 'dart:async';
import 'dart:convert';

import 'package:flutter/services.dart';

import 'package:audio_session/audio_session.dart';
import 'package:just_audio/just_audio.dart';

/// Speaks the recognised phrase using pre-recorded Nepali audio.
///
/// This replaces `flutter_tts`, which could not do the job: `ne-NP` voices are
/// absent from most Android devices and all iOS ones, so device TTS either said
/// nothing or handed the Devanagari to an English engine. It also used to speak
/// the raw GUI label, meaning the phone announced "one. I can't breathe,
/// Medical" — index, category and all.
///
/// A closed six-phrase vocabulary does not need synthesis. Playing files is
/// faster (no engine spin-up), identical on every handset, genuinely offline,
/// and shares its recordings with `scripts/live_demo.py` — the clips in
/// `assets/audio/ne/` are built by `scripts/build_audio.py` and used by both.
class NepaliVoice {
  NepaliVoice._(this._players, this._text, this.missing);

  final Map<String, AudioPlayer> _players;
  final Map<String, String> _text;

  /// The Nepali phrase for [key], or null if there is none — which is the case
  /// for `none` and for any key absent from `assets/phrases.json`.
  ///
  /// Exposed here so the server path, which has no local model and so no
  /// [LocalResult.displayNe], can still show what it just spoke.
  String? textFor(String key) => _text[key];

  /// Class keys with no bundled clip. Empty in a complete build; a UI that can
  /// show it should, because the failure mode is silence, which is otherwise
  /// indistinguishable from the model declining to answer.
  final List<String> missing;

  bool get isEmpty => _players.isEmpty;

  static String assetFor(String key) => 'assets/audio/ne/$key.wav';

  /// Class key -> Nepali text, from `assets/phrases.json` — the same file
  /// `nslr/config.py` and `LocalRecognizer` read, so nothing here duplicates the
  /// phrase list. Keys with a null `ne` (i.e. `none`) are dropped.
  static Future<Map<String, String>> _phraseText() async {
    final raw = await rootBundle.loadString('assets/phrases.json');
    final phrases = (jsonDecode(raw) as Map<String, dynamic>)['phrases']
        as Map<String, dynamic>;
    final out = <String, String>{};
    phrases.forEach((key, value) {
      final ne = (value as Map<String, dynamic>)['ne'];
      if (ne is String) out[key] = ne;
    });
    return out;
  }

  /// Decodes and buffers every clip up front, so the first sign of a session is
  /// no slower to speak than the rest.
  ///
  /// [classNames] normally comes from `model_meta.json`, so this covers exactly
  /// the classes the model can emit. Pass null on the server path, which never
  /// loads a local model and so has no class list of its own — the phrase file's
  /// keys are used instead.
  static Future<NepaliVoice> load(List<String>? classNames) async {
    final text = await _phraseText();
    final keys = classNames ?? text.keys.toList();
    // Without this, iOS defaults to the ambient category and the phrase is
    // SILENT whenever the ringer switch is set to mute — precisely when someone
    // is most likely to be relying on the app. `speech()` also takes transient
    // audio focus on Android, so it ducks music instead of fighting it.
    final session = await AudioSession.instance;
    await session.configure(const AudioSessionConfiguration.speech());

    final players = <String, AudioPlayer>{};
    final missing = <String>[];
    for (final key in keys) {
      // `none` is the open-set negative class — the model declining to answer.
      // It has no clip on disk and is skipped here too, so there is no path by
      // which a rejected sign can be announced aloud.
      if (key == 'none') continue;
      final player = AudioPlayer();
      try {
        await player.setAsset(assetFor(key));
        players[key] = player;
      } catch (_) {
        // A phrase whose recording has not been made yet. It shows text and
        // stays silent rather than blocking the whole app from starting.
        await player.dispose();
        missing.add(key);
      }
    }
    return NepaliVoice._(players, text, missing);
  }

  /// Speaks [key]'s phrase from the beginning. Safe to call for any key.
  ///
  /// Deliberately does not await completion: the caller is finishing a
  /// recognition on the UI thread, and blocking it for the ~3 s of audio would
  /// leave the button disabled while the phrase plays.
  Future<void> play(String key) async {
    final player = _players[key];
    if (player == null) return;
    try {
      // Signs can come faster than a clip finishes; restart rather than
      // overlap, so two phrases are never spoken over each other.
      await player.stop();
      await player.seek(Duration.zero);
      unawaited(player.play());
    } catch (_) {
      // An audio route disappearing mid-play must not surface as a recognition
      // failure — the prediction itself was fine.
    }
  }

  Future<void> stopAll() async {
    for (final player in _players.values) {
      try {
        await player.stop();
      } catch (_) {/* already stopped */}
    }
  }

  Future<void> dispose() async {
    for (final player in _players.values) {
      await player.dispose();
    }
    _players.clear();
  }
}
