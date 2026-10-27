import assert from 'node:assert/strict';
import { test } from 'node:test';
import React from 'react';
import { act, create } from 'react-test-renderer';
import { useAkanadeChat, type UseAkanadeChatOptions, type UseAkanadeChatResult } from '../src/useAkanadeChat';

function fakeFetch(handler: (url: string, init: RequestInit) => { status: number; body: unknown }): typeof fetch {
  return (async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const { status, body } = handler(String(input), init);
    return { ok: status >= 200 && status < 300, status, statusText: String(status), text: async () => JSON.stringify(body) } as Response;
  }) as typeof fetch;
}

// react-test-renderer can't render a hook directly (Rules of Hooks) —
// this harness component calls it and hands the live result out via a
// ref so the test can drive send()/confirm() and assert on state after
// each act() flush.
function Harness({ options, resultRef }: { options: UseAkanadeChatOptions; resultRef: { current: UseAkanadeChatResult | null } }) {
  resultRef.current = useAkanadeChat(options);
  return null;
}

async function renderHook(options: UseAkanadeChatOptions): Promise<{ current: UseAkanadeChatResult | null }> {
  const resultRef: { current: UseAkanadeChatResult | null } = { current: null };
  await act(async () => {
    create(React.createElement(Harness, { options, resultRef }));
  });
  return resultRef;
}

test('send() logs in lazily (no prior token), then sends a chat turn and appends both messages', async () => {
  const calls: string[] = [];
  const fetchImpl = fakeFetch((url, init) => {
    calls.push(url);
    if (url.endsWith('/api/auth/login')) return { status: 200, body: { access_token: 'tok-123' } };
    if (url.endsWith('/api/chat/message')) {
      assert.equal(JSON.parse(init.body as string).message, 'hello');
      return { status: 200, body: { reply: 'hi there', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } };
    }
    throw new Error(`unexpected url ${url}`);
  });

  const ref = await renderHook({ baseUrl: 'https://example.test', email: 'a@b.com', password: 'pw', fetchImpl });

  await act(async () => {
    await ref.current!.send('hello');
  });

  assert.deepEqual(calls, ['https://example.test/api/auth/login', 'https://example.test/api/chat/message']);
  assert.equal(ref.current!.messages.length, 2);
  assert.equal(ref.current!.messages[0].role, 'user');
  assert.equal(ref.current!.messages[0].text, 'hello');
  assert.equal(ref.current!.messages[1].role, 'assistant');
  assert.equal(ref.current!.messages[1].text, 'hi there');
  assert.equal(ref.current!.sending, false);
  assert.equal(ref.current!.error, null);
});

test('a second send() reuses the already-stored access token instead of logging in again', async () => {
  const calls: string[] = [];
  const fetchImpl = fakeFetch((url) => {
    calls.push(url);
    if (url.endsWith('/api/auth/login')) return { status: 200, body: { access_token: 'tok-123' } };
    return { status: 200, body: { reply: 'ok', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } };
  });
  const ref = await renderHook({ baseUrl: 'https://example.test', email: 'a@b.com', password: 'pw', fetchImpl });

  await act(async () => {
    await ref.current!.send('one');
  });
  await act(async () => {
    await ref.current!.send('two');
  });

  const loginCalls = calls.filter((u) => u.endsWith('/api/auth/login'));
  assert.equal(loginCalls.length, 1);
  assert.equal(ref.current!.messages.length, 4);
});

test('send() surfaces a chat error in the error field instead of throwing out of the hook', async () => {
  const fetchImpl = fakeFetch(() => ({ status: 500, body: { detail: 'boom' } }));
  const ref = await renderHook({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });

  await act(async () => {
    await ref.current!.send('hello');
  });

  assert.match(ref.current!.error ?? '', /boom/);
  assert.equal(ref.current!.sending, false);
});

test('a pendingConfirmation reply is reflected in state, and confirm(true) sends "yes"', async () => {
  let turn = 0;
  const sentMessages: string[] = [];
  const fetchImpl = fakeFetch((url, init) => {
    if (url.endsWith('/api/chat/message')) {
      turn += 1;
      sentMessages.push(JSON.parse(init.body as string).message);
      if (turn === 1) {
        return { status: 200, body: { reply: 'Delete the task?', pending_confirmation: true, confirm_prompt: 'Delete the task?', provider: 'groq' } };
      }
      return { status: 200, body: { reply: 'Deleted.', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } };
    }
    throw new Error(`unexpected url ${url}`);
  });
  const ref = await renderHook({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });

  await act(async () => {
    await ref.current!.send('delete my task');
  });
  assert.equal(ref.current!.pendingConfirmation, true);
  assert.equal(ref.current!.confirmPrompt, 'Delete the task?');

  await act(async () => {
    await ref.current!.confirm(true);
  });

  assert.deepEqual(sentMessages, ['delete my task', 'yes']);
  assert.equal(ref.current!.pendingConfirmation, false);
  assert.equal(ref.current!.messages[ref.current!.messages.length - 1].text, 'Deleted.');
});

test('reset() clears messages and confirmation state', async () => {
  const fetchImpl = fakeFetch(() => ({ status: 200, body: { reply: 'hi', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } }));
  const ref = await renderHook({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });

  await act(async () => {
    await ref.current!.send('hello');
  });
  assert.equal(ref.current!.messages.length, 2);

  act(() => {
    ref.current!.reset();
  });

  assert.equal(ref.current!.messages.length, 0);
  assert.equal(ref.current!.pendingConfirmation, false);
});
