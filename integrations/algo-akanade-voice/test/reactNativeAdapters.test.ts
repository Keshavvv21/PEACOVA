import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createReactNativeVoiceAdapter, type ReactNativeTtsModule, type ReactNativeVoiceModule } from '../src/reactNativeAdapters';

function fakeVoiceModule(): ReactNativeVoiceModule & { startedLocale: string | null } {
  const mod = {
    startedLocale: null as string | null,
    onSpeechResults: null,
    onSpeechError: null,
    async start(locale: string) {
      mod.startedLocale = locale;
    },
    async stop() {},
  };
  return mod;
}

function fakeTtsModule(): ReactNativeTtsModule & { spokenText: string | null; trigger: (event: 'tts-finish' | 'tts-error', payload?: unknown) => void } {
  const listeners = new Map<string, Set<(event: unknown) => void>>();
  return {
    spokenText: null,
    speak(text: string) {
      this.spokenText = text;
    },
    addEventListener(event, handler) {
      if (!listeners.has(event)) listeners.set(event, new Set());
      listeners.get(event)!.add(handler);
    },
    removeEventListener(event, handler) {
      listeners.get(event)?.delete(handler);
    },
    trigger(event, payload) {
      listeners.get(event)?.forEach((h) => h(payload));
    },
  };
}

test('recognizeSpeech resolves with the first result and stops listening', async () => {
  const Voice = fakeVoiceModule();
  const Tts = fakeTtsModule();
  const adapter = createReactNativeVoiceAdapter({ Voice, Tts, locale: 'en-GB' });

  const promise = adapter.recognizeSpeech();
  // Simulate the native module firing its callback once recognition finishes.
  Voice.onSpeechResults?.({ value: ['backtest AAPL'] });

  assert.equal(await promise, 'backtest AAPL');
  assert.equal(Voice.startedLocale, 'en-GB');
});

test('recognizeSpeech rejects with the native error message on onSpeechError', async () => {
  const Voice = fakeVoiceModule();
  const Tts = fakeTtsModule();
  const adapter = createReactNativeVoiceAdapter({ Voice, Tts });

  const promise = adapter.recognizeSpeech();
  Voice.onSpeechError?.({ error: { message: 'mic permission denied' } });

  await assert.rejects(promise, /mic permission denied/);
});

test('speak() resolves once the tts-finish event fires and detaches its listeners', async () => {
  const Voice = fakeVoiceModule();
  const Tts = fakeTtsModule();
  const adapter = createReactNativeVoiceAdapter({ Voice, Tts });

  const promise = adapter.speak('hello there');
  assert.equal(Tts.spokenText, 'hello there');
  Tts.trigger('tts-finish');

  await promise; // does not throw/hang
});

test('speak() rejects when tts-error fires', async () => {
  const Voice = fakeVoiceModule();
  const Tts = fakeTtsModule();
  const adapter = createReactNativeVoiceAdapter({ Voice, Tts });

  const promise = adapter.speak('hello there');
  Tts.trigger('tts-error', new Error('tts engine unavailable'));

  await assert.rejects(promise, /tts engine unavailable/);
});
