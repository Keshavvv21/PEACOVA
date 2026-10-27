import 'package:algo_akanade_voice/algo_akanade_voice.dart';
import 'package:test/test.dart';

class FakeSpeechToText implements SpeechToTextLike {
  final bool availableResult;
  void Function(SpeechRecognitionResult)? _onResult;
  FakeSpeechToText({this.availableResult = true});

  @override
  Future<bool> initialize() async => availableResult;

  @override
  Future<void> listen({required void Function(SpeechRecognitionResult result) onResult}) async {
    _onResult = onResult;
  }

  @override
  Future<void> stop() async {}

  void emit(String words, {bool finalResult = true}) {
    _onResult?.call(SpeechRecognitionResult(recognizedWords: words, finalResult: finalResult));
  }
}

class FakeFlutterTts implements FlutterTtsLike {
  String? spokenText;
  void Function()? _onComplete;
  void Function(dynamic)? _onError;

  @override
  Future<dynamic> speak(String text) async {
    spokenText = text;
  }

  @override
  void setCompletionHandler(void Function() handler) => _onComplete = handler;

  @override
  void setErrorHandler(void Function(dynamic message) handler) => _onError = handler;

  void finish() => _onComplete?.call();
  void error(dynamic message) => _onError?.call(message);
}

void main() {
  test('recognizeSpeech resolves with the final recognized words', () async {
    final stt = FakeSpeechToText();
    final tts = FakeFlutterTts();
    final adapter = FlutterVoiceAdapter(speechToText: stt, tts: tts);

    final future = adapter.recognizeSpeech();
    await Future<void>.delayed(Duration.zero);
    stt.emit('backtest AAPL');

    expect(await future, 'backtest AAPL');
  });

  test('recognizeSpeech throws when speech recognition is unavailable on this device', () async {
    final stt = FakeSpeechToText(availableResult: false);
    final tts = FakeFlutterTts();
    final adapter = FlutterVoiceAdapter(speechToText: stt, tts: tts);

    await expectLater(adapter.recognizeSpeech(), throwsStateError);
  });

  test('speak() resolves once the completion handler fires', () async {
    final stt = FakeSpeechToText();
    final tts = FakeFlutterTts();
    final adapter = FlutterVoiceAdapter(speechToText: stt, tts: tts);

    final future = adapter.speak('hello there');
    expect(tts.spokenText, 'hello there');
    tts.finish();

    await future;
  });

  test('speak() rejects when the error handler fires', () async {
    final stt = FakeSpeechToText();
    final tts = FakeFlutterTts();
    final adapter = FlutterVoiceAdapter(speechToText: stt, tts: tts);

    final future = adapter.speak('hello there');
    tts.error('engine unavailable');

    await expectLater(future, throwsA(isA<StateError>()));
  });
}
