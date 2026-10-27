// Adapter for React Native apps, built on the two most common RN
// voice libraries: @react-native-voice/voice (speech-to-text) and
// react-native-tts (text-to-speech). Both are native modules requiring
// their own install + platform linking in a real RN app, so neither is
// a dependency of this package — the already-imported modules are
// passed in, so this file stays installable (and unit-testable with
// fakes) without React Native itself.
//
// Real usage in a React Native app:
//
//   import Voice from '@react-native-voice/voice';
//   import Tts from 'react-native-tts';
//   import { createReactNativeVoiceAdapter } from 'algo-akanade-voice';
//   const voice = createReactNativeVoiceAdapter({ Voice, Tts });
//
// The orchestration logic that calls this adapter (VoiceChatSession)
// is unit-tested; the wiring here against Voice/Tts's documented event
// shapes is also unit-tested against fakes matching those interfaces —
// but it has not run against the real native libraries on a device,
// since no RN runtime/device is available in this sandbox.
import type { VoiceIO } from './voiceChatSession';

export interface ReactNativeVoiceModule {
  start(locale: string): Promise<void>;
  stop(): Promise<void>;
  onSpeechResults: ((event: { value?: string[] }) => void) | null;
  onSpeechError: ((event: { error?: { message?: string } }) => void) | null;
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

export function createReactNativeVoiceAdapter(options: ReactNativeVoiceAdapterOptions): VoiceIO {
  const { Voice, Tts, locale = 'en-US' } = options;

  return {
    recognizeSpeech(): Promise<string> {
      return new Promise((resolve, reject) => {
        Voice.onSpeechResults = (event) => {
          void Voice.stop();
          resolve(event.value?.[0] ?? '');
        };
        Voice.onSpeechError = (event) => {
          void Voice.stop();
          reject(new Error(event.error?.message ?? 'Speech recognition failed.'));
        };
        Voice.start(locale).catch(reject);
      });
    },
    speak(text: string): Promise<void> {
      return new Promise((resolve, reject) => {
        const onFinish = (): void => {
          Tts.removeEventListener('tts-finish', onFinish);
          Tts.removeEventListener('tts-error', onError);
          resolve();
        };
        const onError = (event: unknown): void => {
          Tts.removeEventListener('tts-finish', onFinish);
          Tts.removeEventListener('tts-error', onError);
          reject(event instanceof Error ? event : new Error('Text-to-speech failed.'));
        };
        Tts.addEventListener('tts-finish', onFinish);
        Tts.addEventListener('tts-error', onError);
        Tts.speak(text);
      });
    },
  };
}
