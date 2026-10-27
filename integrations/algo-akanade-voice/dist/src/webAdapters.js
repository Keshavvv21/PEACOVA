"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createWebSpeechAdapter = createWebSpeechAdapter;
function createWebSpeechAdapter(options = {}) {
    const lang = options.lang ?? 'en-US';
    const g = globalThis;
    const SpeechRecognitionCtor = (g.SpeechRecognition ?? g.webkitSpeechRecognition);
    return {
        recognizeSpeech() {
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
        speak(text) {
            const synth = g.speechSynthesis;
            const UtteranceCtor = g.SpeechSynthesisUtterance;
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
