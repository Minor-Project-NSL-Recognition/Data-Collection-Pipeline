import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';

/// Pins the phrase/audio contract that the app cannot check for itself.
///
/// Every failure guarded here is silent at runtime. A class with no phrase entry
/// shows a raw `need_toilet` to a user; a class with no clip simply says nothing,
/// which looks exactly like the model declining to answer; and a clip in the
/// wrong format plays on Android and is mute on iOS. None of them throw.
///
/// Rebuild the clips with:  python scripts/build_audio.py
void main() {
  late Map<String, dynamic> phrases;
  late List<String> classNames;

  setUpAll(() {
    final pf = File('assets/phrases.json');
    if (!pf.existsSync()) fail('missing ${pf.path}');
    phrases = (jsonDecode(pf.readAsStringSync()) as Map<String, dynamic>)['phrases']
        as Map<String, dynamic>;

    final mf = File('assets/models/model_meta.json');
    if (!mf.existsSync()) fail('missing ${mf.path}');
    classNames = ((jsonDecode(mf.readAsStringSync())
        as Map<String, dynamic>)['class_names'] as List).cast<String>();
  });

  test('every class the model can emit has phrase wording', () {
    for (final key in classNames) {
      expect(phrases.containsKey(key), isTrue,
          reason: '$key is in model_meta.json but not phrases.json — the app '
              'would display the raw class key');
    }
  });

  test('every phrase except `none` has Nepali text', () {
    for (final key in classNames) {
      final ne = (phrases[key] as Map<String, dynamic>)['ne'];
      if (key == 'none') {
        // Not an oversight. `none` is the open-set negative class; giving it
        // text or audio is how an app ends up announcing a phrase at someone
        // who did not sign one.
        expect(ne, isNull, reason: '`none` must have no Nepali text');
      } else {
        expect(ne, isA<String>(), reason: '$key has no Nepali text');
        expect((ne as String).trim(), isNotEmpty);
      }
    }
  });

  group('bundled audio', () {
    File clip(String key) => File('assets/audio/ne/$key.wav');

    test('there is no clip for `none`', () {
      expect(clip('none').existsSync(), isFalse,
          reason: 'a rejected sign must stay silent');
    });

    test('every real phrase has a clip', () {
      final missing =
          classNames.where((k) => k != 'none' && !clip(k).existsSync()).toList();
      expect(missing, isEmpty,
          reason: 'no audio for $missing — run: python scripts/build_audio.py');
    });

    test('clips are 48 kHz mono 16-bit PCM', () {
      for (final key in classNames.where((k) => k != 'none')) {
        final f = clip(key);
        if (!f.existsSync()) continue; // reported by the test above
        final fmt = _readWavFmt(f.readAsBytesSync());
        // 1 == WAVE_FORMAT_PCM. winsound in scripts/live_demo.py plays PCM and
        // nothing else, so a compressed WAV would be silent in the demo.
        expect(fmt.audioFormat, 1, reason: '$key is not uncompressed PCM');
        expect(fmt.channels, 1, reason: '$key is not mono');
        expect(fmt.bitsPerSample, 16, reason: '$key is not 16-bit');
        // ffmpeg's loudnorm analyses at 192 kHz and leaks that rate into the
        // output unless -ar is passed, which quietly produced assets 4x larger
        // than needed. Pinned so it cannot come back.
        expect(fmt.sampleRate, 48000, reason: '$key is not 48 kHz');
      }
    });
  });
}

class _WavFmt {
  const _WavFmt(
      this.audioFormat, this.channels, this.sampleRate, this.bitsPerSample);
  final int audioFormat;
  final int channels;
  final int sampleRate;
  final int bitsPerSample;
}

/// Walks the RIFF chunk list to the `fmt ` chunk.
///
/// Not a fixed 44-byte-header read: ffmpeg emits a LIST/INFO chunk before `fmt `
/// often enough that assuming the canonical offset reads metadata as if it were
/// the sample rate.
_WavFmt _readWavFmt(Uint8List bytes) {
  final data = ByteData.sublistView(bytes);
  expect(String.fromCharCodes(bytes.sublist(0, 4)), 'RIFF');
  expect(String.fromCharCodes(bytes.sublist(8, 12)), 'WAVE');

  var offset = 12;
  while (offset + 8 <= bytes.length) {
    final id = String.fromCharCodes(bytes.sublist(offset, offset + 4));
    final size = data.getUint32(offset + 4, Endian.little);
    if (id == 'fmt ') {
      final body = offset + 8;
      return _WavFmt(
        data.getUint16(body, Endian.little),
        data.getUint16(body + 2, Endian.little),
        data.getUint32(body + 4, Endian.little),
        data.getUint16(body + 14, Endian.little),
      );
    }
    offset += 8 + size + (size.isOdd ? 1 : 0); // chunks are word-aligned
  }
  fail('no fmt chunk in WAV');
}
