// A plain Dart client for ALGO AKANADE's existing REST API — the same
// contract documented in
// orchestrator_framework_layer/akanade/README_EXTERNAL_API.md and used
// by the TypeScript twin, ../algo-akanade-connector. This package is
// for a Flutter/Dart app OTHER than tool_layer_akanade/mobile (which
// already has its own, separate client in chat_repository.dart) that
// wants the same integration without reimplementing it from scratch.
import 'dart:convert';
import 'package:http/http.dart' as http;

class AkanadeApiException implements Exception {
  final int status;
  final dynamic body;
  final String message;

  AkanadeApiException(this.status, this.body, this.message);

  @override
  String toString() => message;
}

class ChatReply {
  final String reply;
  final bool pendingConfirmation;
  final String? confirmPrompt;
  final String? provider;

  ChatReply({
    required this.reply,
    required this.pendingConfirmation,
    this.confirmPrompt,
    this.provider,
  });
}

class TrainAlgoAiResult {
  final String domain;
  final String topic;
  final String source;
  final int charsLogged;
  final int ragChunksIndexed;
  final String? jobStatusNote;

  TrainAlgoAiResult({
    required this.domain,
    required this.topic,
    required this.source,
    required this.charsLogged,
    required this.ragChunksIndexed,
    this.jobStatusNote,
  });
}

class RunningTrainingJobs {
  final List<Map<String, dynamic>> jobs;
  final List<String> running;

  RunningTrainingJobs({required this.jobs, required this.running});
}

class AkanadeClient {
  final String baseUrl;
  final http.Client _client;
  String? _accessToken;

  AkanadeClient({required String baseUrl, String? accessToken, http.Client? client})
      : baseUrl = baseUrl.replaceAll(RegExp(r'/+$'), ''),
        _accessToken = accessToken,
        _client = client ?? http.Client();

  String? get accessToken => _accessToken;

  void setAccessToken(String? token) => _accessToken = token;

  /// POST /api/auth/login — stores the returned JWT for subsequent calls.
  Future<String> login(String email, String password) async {
    final data = await _request(
      'POST',
      '/api/auth/login',
      body: {'email': email, 'password': password},
      auth: false,
    );
    _accessToken = data['access_token'] as String;
    return _accessToken!;
  }

  /// POST /api/chat/message — the one endpoint every ALGO AKANADE client
  /// (web, mobile, VS Code) calls for PM Tool, Trading Agent, and
  /// chat-triggered Train Algo AI requests alike.
  Future<ChatReply> chat(String message, String sessionId, {String? provider}) async {
    final data = await _request(
      'POST',
      '/api/chat/message',
      body: {'message': message, 'session_id': sessionId, 'provider': provider},
    );
    return ChatReply(
      reply: data['reply'] as String,
      pendingConfirmation: data['pending_confirmation'] == true,
      confirmPrompt: data['confirm_prompt'] as String?,
      provider: data['provider'] as String?,
    );
  }

  /// Answer a `pendingConfirmation` turn — there is no separate confirm
  /// endpoint, just "yes"/"no" on the same session.
  Future<ChatReply> confirm(String sessionId, bool yes) => chat(yes ? 'yes' : 'no', sessionId);

  /// POST /api/peft-training/link
  Future<TrainAlgoAiResult> trainAlgoAiLink(String domain, String topic, String url) async {
    final data = await _request(
      'POST',
      '/api/peft-training/link',
      body: {'domain': domain, 'topic': topic, 'url': url},
    );
    return _toTrainResult(data);
  }

  /// Convenience wrapper over [trainAlgoAiFile] for plain text that
  /// doesn't already exist as a file.
  Future<TrainAlgoAiResult> trainAlgoAiText(String domain, String topic, String text) {
    final filename = '${topic.isEmpty ? 'note' : topic}.txt';
    return trainAlgoAiFile(domain, topic, filename, utf8.encode(text));
  }

  /// POST /api/peft-training/upload (multipart) — 5MB max, .txt/.md/.csv/.pdf.
  Future<TrainAlgoAiResult> trainAlgoAiFile(
    String domain,
    String topic,
    String filename,
    List<int> fileBytes,
  ) async {
    if (_accessToken == null) {
      throw StateError('Not logged in — call login() first (or pass accessToken in the constructor).');
    }
    final uri = Uri.parse('$baseUrl/api/peft-training/upload');
    final request = http.MultipartRequest('POST', uri)
      ..headers['Authorization'] = 'Bearer $_accessToken'
      ..fields['domain'] = domain
      ..fields['topic'] = topic
      ..files.add(http.MultipartFile.fromBytes('file', fileBytes, filename: filename));
    final streamed = await _client.send(request);
    final res = await http.Response.fromStream(streamed);
    final data = _decodeOrThrow('POST', '/api/peft-training/upload', res);
    return _toTrainResult(data);
  }

  /// GET /api/admin/peft-jobs — Admin-only; returns null (not a throw)
  /// for a non-admin token, matching how the web app hides its badge
  /// silently on a 403.
  Future<RunningTrainingJobs?> runningTrainingJobs() async {
    try {
      final data = await _request('GET', '/api/admin/peft-jobs');
      return RunningTrainingJobs(
        jobs: List<Map<String, dynamic>>.from((data['jobs'] as List?) ?? const []),
        running: List<String>.from((data['running'] as List?) ?? const []),
      );
    } on AkanadeApiException catch (e) {
      if (e.status == 403) return null;
      rethrow;
    }
  }

  Future<Map<String, dynamic>> _request(
    String method,
    String path, {
    Map<String, dynamic>? body,
    bool auth = true,
  }) async {
    final headers = <String, String>{};
    if (auth) {
      if (_accessToken == null) {
        throw StateError('Not logged in — call login() first (or pass accessToken in the constructor).');
      }
      headers['Authorization'] = 'Bearer $_accessToken';
    }
    final uri = Uri.parse('$baseUrl$path');
    http.Response res;
    if (method == 'GET') {
      res = await _client.get(uri, headers: headers);
    } else {
      headers['Content-Type'] = 'application/json';
      res = await _client.post(uri, headers: headers, body: body != null ? jsonEncode(body) : null);
    }
    return _decodeOrThrow(method, path, res);
  }

  Map<String, dynamic> _decodeOrThrow(String method, String path, http.Response res) {
    final Map<String, dynamic> data =
        res.body.isNotEmpty ? jsonDecode(res.body) as Map<String, dynamic> : <String, dynamic>{};
    if (res.statusCode < 200 || res.statusCode >= 300) {
      final detail = data['detail']?.toString() ?? res.reasonPhrase ?? 'error';
      throw AkanadeApiException(
        res.statusCode,
        data,
        'ALGO AKANADE API $method $path failed (${res.statusCode}): $detail',
      );
    }
    return data;
  }

  TrainAlgoAiResult _toTrainResult(Map<String, dynamic> data) {
    return TrainAlgoAiResult(
      domain: data['domain'] as String,
      topic: data['topic'] as String,
      source: data['source'] as String,
      charsLogged: data['chars_logged'] as int,
      ragChunksIndexed: (data['rag_chunks_indexed'] as int?) ?? 0,
      jobStatusNote: data['job_status_note'] as String?,
    );
  }
}
