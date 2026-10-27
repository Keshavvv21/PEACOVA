// Adapter for a real Flutter app, built on the two most common Flutter
// voice packages: speech_to_text (speech-to-text) and flutter_tts
// (text-to-speech). Neither is a dependency of this package — Dart's
// type system is nominal (unlike TypeScript's structural typing), so a
// real SpeechToText/FlutterTts instance needs a small wrapper class
// implementing SpeechToTextLike/FlutterTtsLike below; see this
// package's README for that wrapper, written against each library's
// real, documented API shape.
//
// NOT unit tested against the real plugins in this sandbox — no
// Flutter/device/simulator runtime is available here, and both
// plugins use native platform channels that only work inside a real
// Flutter app. The adapter's own event-wiring logic (this file) IS
// unit tested against fakes matching SpeechToTextLike/FlutterTtsLike
// (see test/flutter_voice_adapter_test.dart).
import 'dart:async';

import 'voice_chat_session.dart';

class SpeechRecognitionResult {
  final String recognizedWords;
  final bool finalResult;

  SpeechRecognitionResult({required this.recognizedWords, required this.finalResult});
}

/// Matches the part of speech_to_text's `SpeechToText` class this adapter uses.
abstract class SpeechToTextLike {
  Future<bool> initialize();
  Future<void> listen({required void Function(SpeechRecognitionResult result) onResult});
  Future<void> stop();
}

/// Matches the part of flutter_tts's `FlutterTts` class this adapter uses.
abstract class FlutterTtsLike {
  Future<dynamic> speak(String text);
  void setCompletionHandler(void Function() handler);
  void setErrorHandler(void Function(dynamic message) handler);
}

class FlutterVoiceAdapter implements VoiceIO {
  final SpeechToTextLike speechToText;
  final FlutterTtsLike tts;

  FlutterVoiceAdapter({required this.speechToText, required this.tts});

  @override
  Future<String> recognizeSpeech() async {
    final available = await speechToText.initialize();
    if (!available) {
      throw StateError('Speech recognition is not available on this device.');
    }
    final completer = Completer<String>();
    await speechToText.listen(
      onResult: (result) {
        if (result.finalResult && !completer.isCompleted) {
          completer.complete(result.recognizedWords);
        }
      },
    );
    final text = await completer.future;
    await speechToText.stop();
    return text;
  }

  @override
  Future<void> speak(String text) {
    final completer = Completer<void>();
    tts.setCompletionHandler(() {
      if (!completer.isCompleted) completer.complete();
    });
    tts.setErrorHandler((message) {
      if (!completer.isCompleted) completer.completeError(StateError('Text-to-speech failed: $message'));
    });
    tts.speak(text);
    return completer.future;
  }
}
