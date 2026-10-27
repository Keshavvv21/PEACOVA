import type { VoiceIO } from './voiceChatSession';
export interface WebSpeechAdapterOptions {
    lang?: string;
}
export declare function createWebSpeechAdapter(options?: WebSpeechAdapterOptions): VoiceIO;
