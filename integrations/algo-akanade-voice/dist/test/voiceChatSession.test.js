"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const strict_1 = __importDefault(require("node:assert/strict"));
const node_test_1 = require("node:test");
const connector_1 = require("@algo-shiki/connector");
const voiceChatSession_1 = require("../src/voiceChatSession");
function fakeShiki(handler) {
    const fetchImpl = (async (input, init = {}) => {
        const { status, body } = handler(String(input), init);
        return { ok: status >= 200 && status < 300, status, statusText: String(status), text: async () => JSON.stringify(body) };
    });
    return new connector_1.ShikiClient({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });
}
function fakeVoice(heard) {
    const spokenLines = [];
    return {
        voice: {
            recognizeSpeech: async () => heard,
            speak: async (text) => {
                spokenLines.push(text);
            },
        },
        spokenLines,
    };
}
(0, node_test_1.test)('stripForSpeech removes an embedded base64 chart image', () => {
    const text = 'Backtest of AAPL: strategy return 12%.\n\n![AAPL equity curve](data:image/png;base64,iVBORw0KGgo=)';
    strict_1.default.equal((0, voiceChatSession_1.stripForSpeech)(text), 'Backtest of AAPL: strategy return 12%.');
});
(0, node_test_1.test)('stripForSpeech removes fenced code blocks and unwraps inline code', () => {
    const text = 'Here you go:\n```python\nprint(1)\n```\nUse `run_backtest()` next.';
    strict_1.default.equal((0, voiceChatSession_1.stripForSpeech)(text), 'Here you go: (code block omitted) Use run_backtest() next.');
});
(0, node_test_1.test)('listenAndReply sends the recognized text to chat() and speaks the (stripped) reply', async () => {
    let sentMessage = '';
    const shiki = fakeShiki((url, init) => {
        strict_1.default.ok(url.endsWith('/api/chat/message'));
        sentMessage = JSON.parse(init.body).message;
        return {
            status: 200,
            body: { reply: 'AAPL returned 12%.\n\n![chart](data:image/png;base64,abc)', pending_confirmation: false, confirm_prompt: null, provider: 'groq' },
        };
    });
    const { voice, spokenLines } = fakeVoice('Backtest AAPL for the last 10 years');
    const session = new voiceChatSession_1.VoiceChatSession({ shiki, sessionId: 's1', voice });
    const result = await session.listenAndReply();
    strict_1.default.equal(sentMessage, 'Backtest AAPL for the last 10 years');
    strict_1.default.equal(result.heard, 'Backtest AAPL for the last 10 years');
    strict_1.default.equal(result.spoken, 'AAPL returned 12%.');
    strict_1.default.deepEqual(spokenLines, ['AAPL returned 12%.']);
});
(0, node_test_1.test)('listenAndReply does not call speak() when nothing speakable remains after stripping', async () => {
    const shiki = fakeShiki(() => ({
        status: 200,
        body: { reply: '![chart](data:image/png;base64,abc)', pending_confirmation: false, confirm_prompt: null, provider: 'groq' },
    }));
    const { voice, spokenLines } = fakeVoice('show me the chart');
    const session = new voiceChatSession_1.VoiceChatSession({ shiki, sessionId: 's1', voice });
    const result = await session.listenAndReply();
    strict_1.default.equal(result.spoken, '');
    strict_1.default.deepEqual(spokenLines, []);
});
(0, node_test_1.test)('listenAndConfirm recognizes an affirmative utterance and sends "yes" as the confirmation', async () => {
    let sentMessage = '';
    const shiki = fakeShiki((url, init) => {
        sentMessage = JSON.parse(init.body).message;
        return { status: 200, body: { reply: 'Done.', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } };
    });
    const { voice } = fakeVoice('yes go ahead');
    const session = new voiceChatSession_1.VoiceChatSession({ shiki, sessionId: 's1', voice });
    await session.listenAndConfirm();
    strict_1.default.equal(sentMessage, 'yes');
});
(0, node_test_1.test)('listenAndConfirm sends "no" for a non-affirmative utterance', async () => {
    let sentMessage = '';
    const shiki = fakeShiki((url, init) => {
        sentMessage = JSON.parse(init.body).message;
        return { status: 200, body: { reply: 'Cancelled.', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } };
    });
    const { voice } = fakeVoice('no, cancel that');
    const session = new voiceChatSession_1.VoiceChatSession({ shiki, sessionId: 's1', voice });
    await session.listenAndConfirm();
    strict_1.default.equal(sentMessage, 'no');
});
