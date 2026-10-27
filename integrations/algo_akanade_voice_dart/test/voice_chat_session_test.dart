import 'dart:convert';

import 'package:algo_akanade_connector/algo_akanade_connector.dart';
import 'package:algo_akanade_voice/algo_akanade_voice.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:test/test.dart';

class FakeVoice implements VoiceIO {
  final String toHear;
  final List<String> spoken = [];
  FakeVoice(this.toHear);

  @override
  Future<String> recognizeSpeech() async => toHear;

  @override
  Future<void> speak(String text) async {
    spoken.add(text);
  }
}

void main() {
  test('stripForSpeech removes an embedded base64 chart image', () {
    const text = 'Backtest of AAPL: strategy return 12%.\n\n![AAPL equity curve](data:image/png;base64,iVBORw0KGgo=)';
    expect(stripForSpeech(text), 'Backtest of AAPL: strategy return 12%.');
  });

  test('stripForSpeech removes fenced code blocks and unwraps inline code', () {
    const text = 'Here you go:\n```python\nprint(1)\n```\nUse `run_backtest()` next.';
    expect(stripForSpeech(text), 'Here you go: (code block omitted) Use run_backtest() next.');
  });

  test('listenAndReply sends the recognized text to chat() and speaks the stripped reply', () async {
    late String sentMessage;
    final client = AkanadeClient(
      baseUrl: 'https://example.test',
      accessToken: 'tok',
      client: MockClient((request) async {
        sentMessage = (jsonDecode((request as http.Request).body) as Map<String, dynamic>)['message'] as String;
        return http.Response(
          jsonEncode({
            'reply': 'AAPL returned 12%.\n\n![chart](data:image/png;base64,abc)',
            'pending_confirmation': false,
            'confirm_prompt': null,
            'provider': 'groq',
          }),
          200,
        );
      }),
    );
    final voice = FakeVoice('Backtest AAPL for the last 10 years');
    final session = VoiceChatSession(akanade: client, sessionId: 's1', voice: voice);

    final result = await session.listenAndReply();

    expect(sentMessage, 'Backtest AAPL for the last 10 years');
    expect(result.spoken, 'AAPL returned 12%.');
    expect(voice.spoken, ['AAPL returned 12%.']);
  });

  test('listenAndReply does not call speak() when nothing speakable remains after stripping', () async {
    final client = AkanadeClient(
      baseUrl: 'https://example.test',
      accessToken: 'tok',
      client: MockClient(
        (request) async => http.Response(
          jsonEncode({
            'reply': '![chart](data:image/png;base64,abc)',
            'pending_confirmation': false,
            'confirm_prompt': null,
            'provider': 'groq',
          }),
          200,
        ),
      ),
    );
    final voice = FakeVoice('show me the chart');
    final session = VoiceChatSession(akanade: client, sessionId: 's1', voice: voice);

    final result = await session.listenAndReply();

    expect(result.spoken, '');
    expect(voice.spoken, isEmpty);
  });

  test('listenAndConfirm sends "yes" for an affirmative utterance', () async {
    late String sentMessage;
    final client = AkanadeClient(
      baseUrl: 'https://example.test',
      accessToken: 'tok',
      client: MockClient((request) async {
        sentMessage = (jsonDecode((request as http.Request).body) as Map<String, dynamic>)['message'] as String;
        return http.Response(
          jsonEncode({'reply': 'Done.', 'pending_confirmation': false, 'confirm_prompt': null, 'provider': 'groq'}),
          200,
        );
      }),
    );
    final voice = FakeVoice('yes go ahead');
    final session = VoiceChatSession(akanade: client, sessionId: 's1', voice: voice);

    await session.listenAndConfirm();

    expect(sentMessage, 'yes');
  });

  test('listenAndConfirm sends "no" for a non-affirmative utterance', () async {
    late String sentMessage;
    final client = AkanadeClient(
      baseUrl: 'https://example.test',
      accessToken: 'tok',
      client: MockClient((request) async {
        sentMessage = (jsonDecode((request as http.Request).body) as Map<String, dynamic>)['message'] as String;
        return http.Response(
          jsonEncode({'reply': 'Cancelled.', 'pending_confirmation': false, 'confirm_prompt': null, 'provider': 'groq'}),
          200,
        );
      }),
    );
    final voice = FakeVoice('no cancel that');
    final session = VoiceChatSession(akanade: client, sessionId: 's1', voice: voice);

    await session.listenAndConfirm();

    expect(sentMessage, 'no');
  });
}
