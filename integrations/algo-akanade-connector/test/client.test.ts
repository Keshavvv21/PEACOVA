import assert from 'node:assert/strict';
import { test } from 'node:test';
import { AkanadeApiError, AkanadeClient } from '../src/index';

interface FakeResponse {
  status: number;
  body: unknown;
}

function fakeFetch(handler: (url: string, init: RequestInit) => FakeResponse): typeof fetch {
  return (async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input);
    const { status, body } = handler(url, init);
    return {
      ok: status >= 200 && status < 300,
      status,
      statusText: String(status),
      text: async () => JSON.stringify(body),
    } as Response;
  }) as typeof fetch;
}

test('login stores the access token and sends it as a bearer header on the next call', async () => {
  const calls: Array<{ url: string; init: RequestInit }> = [];
  const fetchImpl = fakeFetch((url, init) => {
    calls.push({ url, init });
    if (url.endsWith('/api/auth/login')) return { status: 200, body: { access_token: 'tok-123' } };
    if (url.endsWith('/api/chat/message')) {
      return { status: 200, body: { reply: 'hi', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } };
    }
    throw new Error(`unexpected url ${url}`);
  });
  const client = new AkanadeClient({ baseUrl: 'https://example.test', fetchImpl });

  const login = await client.login('a@b.com', 'pw');
  assert.equal(login.accessToken, 'tok-123');

  const reply = await client.chat('hello', 'session-1');
  assert.equal(reply.reply, 'hi');
  assert.equal(reply.provider, 'groq');
  assert.equal(reply.pendingConfirmation, false);

  const chatCall = calls[1];
  assert.equal((chatCall.init.headers as Record<string, string>).Authorization, 'Bearer tok-123');
  assert.deepEqual(JSON.parse(chatCall.init.body as string), {
    message: 'hello',
    session_id: 'session-1',
    provider: null,
  });
});

test('chat without a prior login throws instead of sending an unauthenticated request', async () => {
  const client = new AkanadeClient({
    baseUrl: 'https://example.test',
    fetchImpl: fakeFetch(() => {
      throw new Error('fetch should not be called');
    }),
  });
  await assert.rejects(() => client.chat('hi', 's1'), /Not logged in/);
});

test('a non-2xx response raises AkanadeApiError carrying the status and detail message', async () => {
  const fetchImpl = fakeFetch(() => ({ status: 403, body: { detail: 'Forbidden' } }));
  const client = new AkanadeClient({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });
  await assert.rejects(
    () => client.chat('hi', 's1'),
    (err: unknown) => err instanceof AkanadeApiError && err.status === 403 && /Forbidden/.test(err.message),
  );
});

test('runningTrainingJobs returns null (not a throw) on 403 — non-admin caller', async () => {
  const fetchImpl = fakeFetch(() => ({ status: 403, body: { detail: 'Forbidden' } }));
  const client = new AkanadeClient({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });
  const result = await client.runningTrainingJobs();
  assert.equal(result, null);
});

test('runningTrainingJobs returns jobs/running when authorized', async () => {
  const fetchImpl = fakeFetch(() => ({ status: 200, body: { jobs: [{ id: 'j1' }], running: ['trading'] } }));
  const client = new AkanadeClient({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });
  const result = await client.runningTrainingJobs();
  assert.deepEqual(result, { jobs: [{ id: 'j1' }], running: ['trading'] });
});

test('trainAlgoAiLink posts domain/topic/url and maps the snake_case response to camelCase', async () => {
  const fetchImpl = fakeFetch((url, init) => {
    assert.ok(url.endsWith('/api/peft-training/link'));
    assert.deepEqual(JSON.parse(init.body as string), { domain: 'trading', topic: 'RSI', url: 'https://x' });
    return {
      status: 200,
      body: { domain: 'trading', topic: 'RSI', source: 'https://x', chars_logged: 42, rag_chunks_indexed: 2, job_status_note: null },
    };
  });
  const client = new AkanadeClient({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });
  const result = await client.trainAlgoAiLink('trading', 'RSI', 'https://x');
  assert.deepEqual(result, {
    domain: 'trading',
    topic: 'RSI',
    source: 'https://x',
    charsLogged: 42,
    ragChunksIndexed: 2,
    jobStatusNote: null,
  });
});

test('trainAlgoAiText uploads the text as a multipart file, not JSON', async () => {
  const fetchImpl = fakeFetch((url, init) => {
    assert.ok(url.endsWith('/api/peft-training/upload'));
    assert.ok(init.body instanceof FormData);
    return {
      status: 200,
      body: { domain: 'general', topic: 'note', source: 'note.txt', chars_logged: 5, rag_chunks_indexed: 1, job_status_note: null },
    };
  });
  const client = new AkanadeClient({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });
  const result = await client.trainAlgoAiText('general', 'note', 'hello');
  assert.equal(result.charsLogged, 5);
});
