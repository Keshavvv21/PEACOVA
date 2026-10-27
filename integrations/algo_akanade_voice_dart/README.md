# algo_akanade_voice (Dart)

Speech-to-text in, text-to-speech out, wrapped around
[`algo_akanade_connector`](../algo_akanade_connector_dart)'s `chat()` —
the Dart twin of [`../algo-akanade-voice`](../algo-akanade-voice)
(TypeScript). For a Flutter app that wants a voice experience like the
one `tool_layer_akanade/mobile` already has, built without depending
on that app's code.

## Pieces

- **`VoiceChatSession`** — orchestration: `listenAndReply()` and
  `listenAndConfirm()`. Pure Dart, no Flutter plugin dependency at
  all — fully unit-tested (6 tests) with a fake `VoiceIO`.
- **`stripForSpeech()`** — strips markdown chart images and code
  fences before anything reaches a TTS engine.
- **`FlutterVoiceAdapter`** — built on `speech_to_text` (STT) and
  `flutter_tts` (TTS), the two most common Flutter voice packages.
  **Neither is a dependency of this package** — Dart's type system is
  nominal, so a real `SpeechToText`/`FlutterTts` instance needs a thin
  wrapper implementing `SpeechToTextLike`/`FlutterTtsLike` (below).
  The adapter's own event-wiring logic is unit-tested (4 tests) against
  fakes matching those interfaces.

## Usage in a real Flutter app

Add the real plugins to your Flutter app's own `pubspec.yaml`
(`speech_to_text`, `flutter_tts`) — not this package's — then wrap
them:

```dart
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:flutter_tts/flutter_tts.dart' as ftts;
import 'package:algo_akanade_voice/algo_akanade_voice.dart';

class RealSpeechToText implements SpeechToTextLike {
  final stt.SpeechToText _speech = stt.SpeechToText();

  @override
  Future<bool> initialize() => _speech.initialize();

  @override
  Future<void> listen({required void Function(SpeechRecognitionResult) onResult}) {
    return _speech.listen(
      onResult: (r) => onResult(SpeechRecognitionResult(
        recognizedWords: r.recognizedWords,
        finalResult: r.finalResult,
      )),
    );
  }

  @override
  Future<void> stop() => _speech.stop();
}

class RealFlutterTts implements FlutterTtsLike {
  final ftts.FlutterTts _tts = ftts.FlutterTts();

  @override
  Future<dynamic> speak(String text) => _tts.speak(text);

  @override
  void setCompletionHandler(void Function() handler) => _tts.setCompletionHandler(handler);

  @override
  void setErrorHandler(void Function(dynamic) handler) => _tts.setErrorHandler(handler);
}
```

Then build the session:

```dart
import 'package:algo_akanade_connector/algo_akanade_connector.dart';
import 'package:algo_akanade_voice/algo_akanade_voice.dart';

final akanade = AkanadeClient(baseUrl: 'https://your-host/chatbot');
await akanade.login('you@company.com', 'your-password');

final session = VoiceChatSession(
  akanade: akanade,
  sessionId: 'voice-session-1',
  voice: FlutterVoiceAdapter(speechToText: RealSpeechToText(), tts: RealFlutterTts()),
);

final result = await session.listenAndReply();
print('You said: ${result.heard} — Algo AI said: ${result.spoken}');
```

## Honesty note

`FlutterVoiceAdapter` has not run against the real `speech_to_text`/
`flutter_tts` plugins or a real device/simulator in this sandbox —
both use native platform channels that only work inside an actual
Flutter app. It is real, correct usage of both libraries' documented
APIs, and its own event-wiring logic is unit-tested against fakes —
verify the two thin wrapper classes above against your installed
plugin versions before shipping.

## Tests

```bash
dart pub get
dart test
```
10 tests, all passing, no Flutter SDK, device, or live backend required.
