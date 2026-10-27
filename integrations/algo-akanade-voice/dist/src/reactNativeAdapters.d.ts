import type { VoiceIO } from './voiceChatSession';
export interface ReactNativeVoiceModule {
    start(locale: string): Promise<void>;
    stop(): Promise<void>;
    onSpeechResults: ((event: {
        value?: string[];
    }) => void) | null;
    onSpeechError: ((event: {
        error?: {
            message?: string;
        };
    }) => void) | null;
}
export interface ReactNativeTtsModule {
    speak(text: string): void;
    addEventListener(event: 'tts-finish' | 'tts-cancel' | 'tts-error', handler: (event: unknown) => void): void;
    removeEventListener(event: 'tts-finish' | 'tts-cancel' | 'tts-error', handler: (event: unknown) => void): void;
}
export interface ReactNativeVoiceAdapterOptions {
    Voice: ReactNativeVoiceModule;
    Tts: ReactNativeTtsModule;
    locale?: string;
}
export declare function createReactNativeVoiceAdapter(options: ReactNativeVoiceAdapterOptions): VoiceIO;
