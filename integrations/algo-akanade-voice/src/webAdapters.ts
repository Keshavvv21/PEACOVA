// Browser adapter using the Web Speech API (SpeechRecognition +
// speechSynthesis) — both are real, shipping browser APIs. Not unit
// tested in this sandbox (no browser environment here); this is
// standard, documented usage of those two APIs, not exploratory code.
// SpeechRecognition support varies by browser (works unprefixed in
// Chrome/Edge; Safari and Firefox have partial/no support as of this
// writing) — check for it and fall back to a text input in your UI.
import type { VoiceIO } from './voiceChatSession';

export interface WebSpeechAdapterOptions {
  lang?: string;
}

export function createWebSpeechAdapter(options: WebSpeechAdapterOptions = {}): VoiceIO {
  const lang = options.lang ?? 'en-US';
  const g = globalThis as unknown as Record<string, unknown>;
  const SpeechRecognitionCtor = (g.SpeechRecognition ?? g.webkitSpeechRecognition) as
    | (new () => {
        lang: string;
        interimResults: boolean;
        maxAlternatives: number;
        onresult: ((event: { results: { 0: { 0: { transcript: string } } } }) => void) | null;
        onerror: ((event: { error: string }) => void) | null;
        start(): void;
      })
    | undefined;

  return {
    recognizeSpeech(): Promise<string> {
      if (!SpeechRecognitionCtor) {
        return Promise.reject(new Error('SpeechRecognition is not available in this browser.'));
      }
      return new Promise((resolve, reject) => {
        const recognizer = new SpeechRecognitionCtor();
        recognizer.lang = lang;
        recognizer.interimResults = false;
        recognizer.maxAlternatives = 1;
        recognizer.onresult = (event) => resolve(event.results[0][0].transcript);
        recognizer.onerror = (event) => reject(new Error(`Speech recognition error: ${event.error}`));
        recognizer.start();
      });
    },
    speak(text: string): Promise<void> {
      const synth = g.speechSynthesis as { speak(u: unknown): void } | undefined;
      const UtteranceCtor = g.SpeechSynthesisUtterance as
        | (new (text: string) => { lang: string; onend: (() => void) | null; onerror: ((event: { error: string }) => void) | null })
        | undefined;
      if (!synth || !UtteranceCtor) {
        return Promise.reject(new Error('speechSynthesis is not available in this browser.'));
      }
      return new Promise((resolve, reject) => {
        const utterance = new UtteranceCtor(text);
        utterance.lang = lang;
        utterance.onend = () => resolve();
        utterance.onerror = (event) => reject(new Error(`Speech synthesis error: ${event.error}`));
        synth.speak(utterance);
      });
    },
  };
}
