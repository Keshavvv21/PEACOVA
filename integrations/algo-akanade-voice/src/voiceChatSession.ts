// Orchestration only — no platform speech APIs here, so this file is
// fully unit-testable with a fake VoiceIO (see test/voiceChatSession.test.ts).
// Real platform wiring lives in webAdapters.ts (browser) and
// reactNativeAdapters.ts (React Native).
import type { ChatReply, AkanadeClient } from '@algo-akanade/connector';

export interface VoiceIO {
  /** Listens for one utterance and resolves with the recognized text. */
  recognizeSpeech(): Promise<string>;
  /** Speaks `text` aloud, resolving once playback finishes. */
  speak(text: string): Promise<void>;
}

export interface VoiceChatSessionOptions {
  akanade: AkanadeClient;
  sessionId: string;
  voice: VoiceIO;
  /** Same meaning as AkanadeClient.chat()'s provider param — null/omitted for auto-routing. */
  provider?: string | null;
}

export interface VoiceTurnResult {
  heard: string;
  reply: ChatReply;
  /** What was actually spoken — reply.reply with images/code stripped (see stripForSpeech). Empty string if nothing speakable remained. */
  spoken: string;
}

/**
 * Chat replies can embed a base64 PNG chart (the Trading Agent's
 * backtest charts — see trading_tools.py's chart_base64) or fenced
 * code blocks. Neither is speakable, so this strips them before
 * anything is passed to a text-to-speech engine.
 */
export function stripForSpeech(text: string): string {
  return text
    .replace(/!\[[^\]]*\]\(data:[^)]+\)/g, '')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
    .replace(/```[\s\S]*?```/g, ' (code block omitted) ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\s+/g, ' ')
    .trim();
}

export class VoiceChatSession {
  private akanade: AkanadeClient;
  private sessionId: string;
  private voice: VoiceIO;
  private provider: string | null;

  constructor(options: VoiceChatSessionOptions) {
    this.akanade = options.akanade;
    this.sessionId = options.sessionId;
    this.voice = options.voice;
    this.provider = options.provider ?? null;
  }

  /** Listens for one spoken turn, sends it to ALGO AKANADE, and speaks the reply back. */
  async listenAndReply(): Promise<VoiceTurnResult> {
    const heard = await this.voice.recognizeSpeech();
    const reply = await this.akanade.chat(heard, this.sessionId, this.provider);
    const spoken = stripForSpeech(reply.reply);
    if (spoken) await this.voice.speak(spoken);
    return { heard, reply, spoken };
  }

  /** For a pendingConfirmation turn — listens for a yes/no-shaped utterance and sends it as the confirmation. */
  async listenAndConfirm(): Promise<VoiceTurnResult> {
    const heard = await this.voice.recognizeSpeech();
    const yes = /^\s*(yes|yeah|yep|confirm|sure|go ahead)\b/i.test(heard);
    const reply = await this.akanade.confirm(this.sessionId, yes);
    const spoken = stripForSpeech(reply.reply);
    if (spoken) await this.voice.speak(spoken);
    return { heard, reply, spoken };
  }
}
