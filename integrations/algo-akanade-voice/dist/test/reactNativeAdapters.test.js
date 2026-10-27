"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const strict_1 = __importDefault(require("node:assert/strict"));
const node_test_1 = require("node:test");
const reactNativeAdapters_1 = require("../src/reactNativeAdapters");
function fakeVoiceModule() {
    const mod = {
        startedLocale: null,
        onSpeechResults: null,
        onSpeechError: null,
        async start(locale) {
            mod.startedLocale = locale;
        },
        async stop() { },
    };
    return mod;
}
function fakeTtsModule() {
    const listeners = new Map();
    return {
        spokenText: null,
        speak(text) {
            this.spokenText = text;
        },
        addEventListener(event, handler) {
            if (!listeners.has(event))
                listeners.set(event, new Set());
            listeners.get(event).add(handler);
        },
        removeEventListener(event, handler) {
            listeners.get(event)?.delete(handler);
        },
        trigger(event, payload) {
            listeners.get(event)?.forEach((h) => h(payload));
        },
    };
}
(0, node_test_1.test)('recognizeSpeech resolves with the first result and stops listening', async () => {
    const Voice = fakeVoiceModule();
    const Tts = fakeTtsModule();
    const adapter = (0, reactNativeAdapters_1.createReactNativeVoiceAdapter)({ Voice, Tts, locale: 'en-GB' });
    const promise = adapter.recognizeSpeech();
    // Simulate the native module firing its callback once recognition finishes.
    Voice.onSpeechResults?.({ value: ['backtest AAPL'] });
    strict_1.default.equal(await promise, 'backtest AAPL');
    strict_1.default.equal(Voice.startedLocale, 'en-GB');
});
(0, node_test_1.test)('recognizeSpeech rejects with the native error message on onSpeechError', async () => {
    const Voice = fakeVoiceModule();
    const Tts = fakeTtsModule();
    const adapter = (0, reactNativeAdapters_1.createReactNativeVoiceAdapter)({ Voice, Tts });
    const promise = adapter.recognizeSpeech();
    Voice.onSpeechError?.({ error: { message: 'mic permission denied' } });
    await strict_1.default.rejects(promise, /mic permission denied/);
});
(0, node_test_1.test)('speak() resolves once the tts-finish event fires and detaches its listeners', async () => {
    const Voice = fakeVoiceModule();
    const Tts = fakeTtsModule();
    const adapter = (0, reactNativeAdapters_1.createReactNativeVoiceAdapter)({ Voice, Tts });
    const promise = adapter.speak('hello there');
    strict_1.default.equal(Tts.spokenText, 'hello there');
    Tts.trigger('tts-finish');
    await promise; // does not throw/hang
});
(0, node_test_1.test)('speak() rejects when tts-error fires', async () => {
    const Voice = fakeVoiceModule();
    const Tts = fakeTtsModule();
    const adapter = (0, reactNativeAdapters_1.createReactNativeVoiceAdapter)({ Voice, Tts });
    const promise = adapter.speak('hello there');
    Tts.trigger('tts-error', new Error('tts engine unavailable'));
    await strict_1.default.rejects(promise, /tts engine unavailable/);
});
