import type { ChatReply, ShikiClient } from '@algo-shiki/connector';
export interface VoiceIO {
    /** Listens for one utterance and resolves with the recognized text. */
    recognizeSpeech(): Promise<string>;
    /** Speaks `text` aloud, resolving once playback finishes. */
    speak(text: string): Promise<void>;
}
export interface VoiceChatSessionOptions {
    shiki: ShikiClient;
    sessionId: string;
    voice: VoiceIO;
    /** Same meaning as ShikiClient.chat()'s provider param — null/omitted for auto-routing. */
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
export declare function stripForSpeech(text: string): string;
export declare class VoiceChatSession {
    private shiki;
    private sessionId;
    private voice;
    private provider;
    constructor(options: VoiceChatSessionOptions);
    /** Listens for one spoken turn, sends it to ALGO SHIKI, and speaks the reply back. */
    listenAndReply(): Promise<VoiceTurnResult>;
    /** For a pendingConfirmation turn — listens for a yes/no-shaped utterance and sends it as the confirmation. */
    listenAndConfirm(): Promise<VoiceTurnResult>;
}
