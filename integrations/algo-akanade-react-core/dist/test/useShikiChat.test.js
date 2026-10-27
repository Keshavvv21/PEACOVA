"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const strict_1 = __importDefault(require("node:assert/strict"));
const node_test_1 = require("node:test");
const react_1 = __importDefault(require("react"));
const react_test_renderer_1 = require("react-test-renderer");
const useShikiChat_1 = require("../src/useShikiChat");
function fakeFetch(handler) {
    return (async (input, init = {}) => {
        const { status, body } = handler(String(input), init);
        return { ok: status >= 200 && status < 300, status, statusText: String(status), text: async () => JSON.stringify(body) };
    });
}
// react-test-renderer can't render a hook directly (Rules of Hooks) —
// this harness component calls it and hands the live result out via a
// ref so the test can drive send()/confirm() and assert on state after
// each act() flush.
function Harness({ options, resultRef }) {
    resultRef.current = (0, useShikiChat_1.useShikiChat)(options);
    return null;
}
async function renderHook(options) {
    const resultRef = { current: null };
    await (0, react_test_renderer_1.act)(async () => {
        (0, react_test_renderer_1.create)(react_1.default.createElement(Harness, { options, resultRef }));
    });
    return resultRef;
}
(0, node_test_1.test)('send() logs in lazily (no prior token), then sends a chat turn and appends both messages', async () => {
    const calls = [];
    const fetchImpl = fakeFetch((url, init) => {
        calls.push(url);
        if (url.endsWith('/api/auth/login'))
            return { status: 200, body: { access_token: 'tok-123' } };
        if (url.endsWith('/api/chat/message')) {
            strict_1.default.equal(JSON.parse(init.body).message, 'hello');
            return { status: 200, body: { reply: 'hi there', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } };
        }
        throw new Error(`unexpected url ${url}`);
    });
    const ref = await renderHook({ baseUrl: 'https://example.test', email: 'a@b.com', password: 'pw', fetchImpl });
    await (0, react_test_renderer_1.act)(async () => {
        await ref.current.send('hello');
    });
    strict_1.default.deepEqual(calls, ['https://example.test/api/auth/login', 'https://example.test/api/chat/message']);
    strict_1.default.equal(ref.current.messages.length, 2);
    strict_1.default.equal(ref.current.messages[0].role, 'user');
    strict_1.default.equal(ref.current.messages[0].text, 'hello');
    strict_1.default.equal(ref.current.messages[1].role, 'assistant');
    strict_1.default.equal(ref.current.messages[1].text, 'hi there');
    strict_1.default.equal(ref.current.sending, false);
    strict_1.default.equal(ref.current.error, null);
});
(0, node_test_1.test)('a second send() reuses the already-stored access token instead of logging in again', async () => {
    const calls = [];
    const fetchImpl = fakeFetch((url) => {
        calls.push(url);
        if (url.endsWith('/api/auth/login'))
            return { status: 200, body: { access_token: 'tok-123' } };
        return { status: 200, body: { reply: 'ok', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } };
    });
    const ref = await renderHook({ baseUrl: 'https://example.test', email: 'a@b.com', password: 'pw', fetchImpl });
    await (0, react_test_renderer_1.act)(async () => {
        await ref.current.send('one');
    });
    await (0, react_test_renderer_1.act)(async () => {
        await ref.current.send('two');
    });
    const loginCalls = calls.filter((u) => u.endsWith('/api/auth/login'));
    strict_1.default.equal(loginCalls.length, 1);
    strict_1.default.equal(ref.current.messages.length, 4);
});
(0, node_test_1.test)('send() surfaces a chat error in the error field instead of throwing out of the hook', async () => {
    const fetchImpl = fakeFetch(() => ({ status: 500, body: { detail: 'boom' } }));
    const ref = await renderHook({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });
    await (0, react_test_renderer_1.act)(async () => {
        await ref.current.send('hello');
    });
    strict_1.default.match(ref.current.error ?? '', /boom/);
    strict_1.default.equal(ref.current.sending, false);
});
(0, node_test_1.test)('a pendingConfirmation reply is reflected in state, and confirm(true) sends "yes"', async () => {
    let turn = 0;
    const sentMessages = [];
    const fetchImpl = fakeFetch((url, init) => {
        if (url.endsWith('/api/chat/message')) {
            turn += 1;
            sentMessages.push(JSON.parse(init.body).message);
            if (turn === 1) {
                return { status: 200, body: { reply: 'Delete the task?', pending_confirmation: true, confirm_prompt: 'Delete the task?', provider: 'groq' } };
            }
            return { status: 200, body: { reply: 'Deleted.', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } };
        }
        throw new Error(`unexpected url ${url}`);
    });
    const ref = await renderHook({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });
    await (0, react_test_renderer_1.act)(async () => {
        await ref.current.send('delete my task');
    });
    strict_1.default.equal(ref.current.pendingConfirmation, true);
    strict_1.default.equal(ref.current.confirmPrompt, 'Delete the task?');
    await (0, react_test_renderer_1.act)(async () => {
        await ref.current.confirm(true);
    });
    strict_1.default.deepEqual(sentMessages, ['delete my task', 'yes']);
    strict_1.default.equal(ref.current.pendingConfirmation, false);
    strict_1.default.equal(ref.current.messages[ref.current.messages.length - 1].text, 'Deleted.');
});
(0, node_test_1.test)('reset() clears messages and confirmation state', async () => {
    const fetchImpl = fakeFetch(() => ({ status: 200, body: { reply: 'hi', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } }));
    const ref = await renderHook({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });
    await (0, react_test_renderer_1.act)(async () => {
        await ref.current.send('hello');
    });
    strict_1.default.equal(ref.current.messages.length, 2);
    (0, react_test_renderer_1.act)(() => {
        ref.current.reset();
    });
    strict_1.default.equal(ref.current.messages.length, 0);
    strict_1.default.equal(ref.current.pendingConfirmation, false);
});
