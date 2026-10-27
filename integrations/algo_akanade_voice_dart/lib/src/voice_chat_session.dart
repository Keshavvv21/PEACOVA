// Orchestration only — no Flutter plugin code here, so this file is
// fully unit-testable with a fake VoiceIO. Real platform wiring lives
// in flutter_voice_adapter.dart. This is the Dart twin of
// ../algo-akanade-voice (TypeScript)'s voiceChatSession.ts.
import 'package:algo_akanade_connector/algo_akanade_connector.dart';

abstract class VoiceIO {
  /// Listens for one utterance and resolves with the recognized text.
  Future<String> recognizeSpeech();

  /// Speaks [text] aloud, resolving once playback finishes.
  Future<void> speak(String text);
}

/// Chat replies can embed a base64 PNG chart (the Trading Agent's
/// backtest charts) or fenced code blocks. Neither is speakable, so
/// this strips them before anything is passed to a text-to-speech
/// engine.
String stripForSpeech(String text) {
  var result = text;
  result = result.replaceAll(RegExp(r'!\[[^\]]*\]\(data:[^)]+\)'), '');
  result = result.replaceAll(RegExp(r'!\[[^\]]*\]\([^)]*\)'), '');
  result = result.replaceAll(RegExp(r'```[\s\S]*?```'), ' (code block omitted) ');
  result = result.replaceAllMapped(RegExp(r'`([^`]+)`'), (m) => m.group(1) ?? '');
  result = result.replaceAll(RegExp(r'\s+'), ' ').trim();
  return result;
}

class VoiceTurnResult {
  final String heard;
  final ChatReply reply;
  final String spoken;

  VoiceTurnResult({required this.heard, required this.reply, required this.spoken});
}

class VoiceChatSession {
  final AkanadeClient akanade;
  final String sessionId;
  final VoiceIO voice;
  final String? provider;

  VoiceChatSession({
    required this.akanade,
    required this.sessionId,
    required this.voice,
    this.provider,
  });

  /// Listens for one spoken turn, sends it to ALGO AKANADE, and speaks the reply back.
  Future<VoiceTurnResult> listenAndReply() async {
    final heard = await voice.recognizeSpeech();
    final reply = await akanade.chat(heard, sessionId, provider: provider);
    final spoken = stripForSpeech(reply.reply);
    if (spoken.isNotEmpty) await voice.speak(spoken);
    return VoiceTurnResult(heard: heard, reply: reply, spoken: spoken);
  }

  /// For a pendingConfirmation turn — listens for a yes/no-shaped utterance and sends it as the confirmation.
  Future<VoiceTurnResult> listenAndConfirm() async {
    final heard = await voice.recognizeSpeech();
    final yes = RegExp(r'^\s*(yes|yeah|yep|confirm|sure|go ahead)\b', caseSensitive: false).hasMatch(heard);
    final reply = await akanade.confirm(sessionId, yes);
    final spoken = stripForSpeech(reply.reply);
    if (spoken.isNotEmpty) await voice.speak(spoken);
    return VoiceTurnResult(heard: heard, reply: reply, spoken: spoken);
  }
}
