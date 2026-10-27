"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const strict_1 = __importDefault(require("node:assert/strict"));
const node_test_1 = require("node:test");
const index_1 = require("../src/index");
function fakeFetch(handler) {
    return (async (input, init = {}) => {
        const url = String(input);
        const { status, body } = handler(url, init);
        return {
            ok: status >= 200 && status < 300,
            status,
            statusText: String(status),
            text: async () => JSON.stringify(body),
        };
    });
}
(0, node_test_1.test)('login stores the access token and sends it as a bearer header on the next call', async () => {
    const calls = [];
    const fetchImpl = fakeFetch((url, init) => {
        calls.push({ url, init });
        if (url.endsWith('/api/auth/login'))
            return { status: 200, body: { access_token: 'tok-123' } };
        if (url.endsWith('/api/chat/message')) {
            return { status: 200, body: { reply: 'hi', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } };
        }
        throw new Error(`unexpected url ${url}`);
    });
    const client = new index_1.ShikiClient({ baseUrl: 'https://example.test', fetchImpl });
    const login = await client.login('a@b.com', 'pw');
    strict_1.default.equal(login.accessToken, 'tok-123');
    const reply = await client.chat('hello', 'session-1');
    strict_1.default.equal(reply.reply, 'hi');
    strict_1.default.equal(reply.provider, 'groq');
    strict_1.default.equal(reply.pendingConfirmation, false);
    const chatCall = calls[1];
    strict_1.default.equal(chatCall.init.headers.Authorization, 'Bearer tok-123');
    strict_1.default.deepEqual(JSON.parse(chatCall.init.body), {
        message: 'hello',
        session_id: 'session-1',
        provider: null,
    });
});
(0, node_test_1.test)('chat without a prior login throws instead of sending an unauthenticated request', async () => {
    const client = new index_1.ShikiClient({
        baseUrl: 'https://example.test',
        fetchImpl: fakeFetch(() => {
            throw new Error('fetch should not be called');
        }),
    });
    await strict_1.default.rejects(() => client.chat('hi', 's1'), /Not logged in/);
});
(0, node_test_1.test)('a non-2xx response raises ShikiApiError carrying the status and detail message', async () => {
    const fetchImpl = fakeFetch(() => ({ status: 403, body: { detail: 'Forbidden' } }));
    const client = new index_1.ShikiClient({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });
    await strict_1.default.rejects(() => client.chat('hi', 's1'), (err) => err instanceof index_1.ShikiApiError && err.status === 403 && /Forbidden/.test(err.message));
});
(0, node_test_1.test)('runningTrainingJobs returns null (not a throw) on 403 — non-admin caller', async () => {
    const fetchImpl = fakeFetch(() => ({ status: 403, body: { detail: 'Forbidden' } }));
    const client = new index_1.ShikiClient({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });
    const result = await client.runningTrainingJobs();
    strict_1.default.equal(result, null);
});
(0, node_test_1.test)('runningTrainingJobs returns jobs/running when authorized', async () => {
    const fetchImpl = fakeFetch(() => ({ status: 200, body: { jobs: [{ id: 'j1' }], running: ['trading'] } }));
    const client = new index_1.ShikiClient({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });
    const result = await client.runningTrainingJobs();
    strict_1.default.deepEqual(result, { jobs: [{ id: 'j1' }], running: ['trading'] });
});
(0, node_test_1.test)('trainAlgoAiLink posts domain/topic/url and maps the snake_case response to camelCase', async () => {
    const fetchImpl = fakeFetch((url, init) => {
        strict_1.default.ok(url.endsWith('/api/peft-training/link'));
        strict_1.default.deepEqual(JSON.parse(init.body), { domain: 'trading', topic: 'RSI', url: 'https://x' });
        return {
            status: 200,
            body: { domain: 'trading', topic: 'RSI', source: 'https://x', chars_logged: 42, rag_chunks_indexed: 2, job_status_note: null },
        };
    });
    const client = new index_1.ShikiClient({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });
    const result = await client.trainAlgoAiLink('trading', 'RSI', 'https://x');
    strict_1.default.deepEqual(result, {
        domain: 'trading',
        topic: 'RSI',
        source: 'https://x',
        charsLogged: 42,
        ragChunksIndexed: 2,
        jobStatusNote: null,
    });
});
(0, node_test_1.test)('trainAlgoAiText uploads the text as a multipart file, not JSON', async () => {
    const fetchImpl = fakeFetch((url, init) => {
        strict_1.default.ok(url.endsWith('/api/peft-training/upload'));
        strict_1.default.ok(init.body instanceof FormData);
        return {
            status: 200,
            body: { domain: 'general', topic: 'note', source: 'note.txt', chars_logged: 5, rag_chunks_indexed: 1, job_status_note: null },
        };
    });
    const client = new index_1.ShikiClient({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });
    const result = await client.trainAlgoAiText('general', 'note', 'hello');
    strict_1.default.equal(result.charsLogged, 5);
});
