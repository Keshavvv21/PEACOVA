import 'dart:convert';

import 'package:algo_akanade_connector/algo_akanade_connector.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:test/test.dart';

void main() {
  test('login stores the token and sends it as a bearer header on the next call', () async {
    final calls = <http.BaseRequest>[];
    final client = MockClient((request) async {
      calls.add(request);
      if (request.url.path.endsWith('/api/auth/login')) {
        return http.Response(jsonEncode({'access_token': 'tok-123'}), 200);
      }
      if (request.url.path.endsWith('/api/chat/message')) {
        return http.Response(
          jsonEncode({'reply': 'hi', 'pending_confirmation': false, 'confirm_prompt': null, 'provider': 'groq'}),
          200,
        );
      }
      throw Exception('unexpected url ${request.url}');
    });
    final akanade = AkanadeClient(baseUrl: 'https://example.test', client: client);

    final token = await akanade.login('a@b.com', 'pw');
    expect(token, 'tok-123');
    expect(akanade.accessToken, 'tok-123');

    final reply = await akanade.chat('hello', 'session-1');
    expect(reply.reply, 'hi');
    expect(reply.provider, 'groq');
    expect(reply.pendingConfirmation, isFalse);

    final chatCall = calls[1] as http.Request;
    expect(chatCall.headers['Authorization'], 'Bearer tok-123');
    expect(jsonDecode(chatCall.body), {'message': 'hello', 'session_id': 'session-1', 'provider': null});
  });

  test('chat without a prior login throws instead of sending an unauthenticated request', () async {
    final client = MockClient((request) async {
      throw Exception('should not be called');
    });
    final akanade = AkanadeClient(baseUrl: 'https://example.test', client: client);
    expect(() => akanade.chat('hi', 's1'), throwsStateError);
  });

  test('a non-2xx response raises AkanadeApiException carrying the status and detail message', () async {
    final client = MockClient((request) async {
      return http.Response(jsonEncode({'detail': 'Forbidden'}), 403);
    });
    final akanade = AkanadeClient(baseUrl: 'https://example.test', accessToken: 'tok', client: client);
    await expectLater(
      () => akanade.chat('hi', 's1'),
      throwsA(
        isA<AkanadeApiException>()
            .having((e) => e.status, 'status', 403)
            .having((e) => e.message, 'message', contains('Forbidden')),
      ),
    );
  });

  test('runningTrainingJobs returns null (not a throw) on 403 — non-admin caller', () async {
    final client = MockClient((request) async => http.Response(jsonEncode({'detail': 'Forbidden'}), 403));
    final akanade = AkanadeClient(baseUrl: 'https://example.test', accessToken: 'tok', client: client);
    final result = await akanade.runningTrainingJobs();
    expect(result, isNull);
  });

  test('runningTrainingJobs returns jobs/running when authorized', () async {
    final client = MockClient((request) async {
      return http.Response(
        jsonEncode({
          'jobs': [
            {'id': 'j1'}
          ],
          'running': ['trading']
        }),
        200,
      );
    });
    final akanade = AkanadeClient(baseUrl: 'https://example.test', accessToken: 'tok', client: client);
    final result = await akanade.runningTrainingJobs();
    expect(result, isNotNull);
    expect(result!.running, ['trading']);
    expect(result.jobs.single['id'], 'j1');
  });

  test('trainAlgoAiLink posts domain/topic/url and maps the snake_case response', () async {
    final client = MockClient((request) async {
      expect(request.url.path, endsWith('/api/peft-training/link'));
      final sent = jsonDecode((request as http.Request).body);
      expect(sent, {'domain': 'trading', 'topic': 'RSI', 'url': 'https://x'});
      return http.Response(
        jsonEncode({
          'domain': 'trading',
          'topic': 'RSI',
          'source': 'https://x',
          'chars_logged': 42,
          'rag_chunks_indexed': 2,
          'job_status_note': null,
        }),
        200,
      );
    });
    final akanade = AkanadeClient(baseUrl: 'https://example.test', accessToken: 'tok', client: client);
    final result = await akanade.trainAlgoAiLink('trading', 'RSI', 'https://x');
    expect(result.domain, 'trading');
    expect(result.charsLogged, 42);
    expect(result.ragChunksIndexed, 2);
    expect(result.jobStatusNote, isNull);
  });

  test('trainAlgoAiText uploads the text as a multipart file, not JSON', () async {
    final client = MockClient((request) async {
      expect(request.url.path, endsWith('/api/peft-training/upload'));
      // MockClient finalizes MultipartRequest into a plain Request, reading
      // the whole encoded multipart body — assert on that raw body instead
      // of the original request object, which no longer exists by here.
      final plain = request as http.Request;
      expect(plain.headers['content-type'], contains('multipart/form-data'));
      expect(plain.body, contains('name="domain"'));
      expect(plain.body, contains('general'));
      expect(plain.body, contains('filename="note.txt"'));
      return http.Response(
        jsonEncode({
          'domain': 'general',
          'topic': 'note',
          'source': 'note.txt',
          'chars_logged': 5,
          'rag_chunks_indexed': 1,
          'job_status_note': null,
        }),
        200,
      );
    });
    final akanade = AkanadeClient(baseUrl: 'https://example.test', accessToken: 'tok', client: client);
    final result = await akanade.trainAlgoAiText('general', 'note', 'hello');
    expect(result.charsLogged, 5);
  });
}
