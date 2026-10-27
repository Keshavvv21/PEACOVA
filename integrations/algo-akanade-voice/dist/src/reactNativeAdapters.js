"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createReactNativeVoiceAdapter = createReactNativeVoiceAdapter;
function createReactNativeVoiceAdapter(options) {
    const { Voice, Tts, locale = 'en-US' } = options;
    return {
        recognizeSpeech() {
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
        speak(text) {
            return new Promise((resolve, reject) => {
                const onFinish = () => {
                    Tts.removeEventListener('tts-finish', onFinish);
                    Tts.removeEventListener('tts-error', onError);
                    resolve();
                };
                const onError = (event) => {
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
