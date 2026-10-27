import assert from 'node:assert/strict';
import { test } from 'node:test';
import { AkanadeClient } from '@algo-akanade/connector';
import { stripForSpeech, VoiceChatSession, type VoiceIO } from '../src/voiceChatSession';

function fakeAkanade(handler: (url: string, init: RequestInit) => { status: number; body: unknown }): AkanadeClient {
  const fetchImpl = (async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const { status, body } = handler(String(input), init);
    return { ok: status >= 200 && status < 300, status, statusText: String(status), text: async () => JSON.stringify(body) } as Response;
  }) as typeof fetch;
  return new AkanadeClient({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });
}

function fakeVoice(heard: string): { voice: VoiceIO; spokenLines: string[] } {
  const spokenLines: string[] = [];
  return {
    voice: {
      recognizeSpeech: async () => heard,
      speak: async (text: string) => {
        spokenLines.push(text);
      },
    },
    spokenLines,
  };
}

test('stripForSpeech removes an embedded base64 chart image', () => {
  const text = 'Backtest of AAPL: strategy return 12%.\n\n![AAPL equity curve](data:image/png;base64,iVBORw0KGgo=)';
  assert.equal(stripForSpeech(text), 'Backtest of AAPL: strategy return 12%.');
});

test('stripForSpeech removes fenced code blocks and unwraps inline code', () => {
  const text = 'Here you go:\n```python\nprint(1)\n```\nUse `run_backtest()` next.';
  assert.equal(stripForSpeech(text), 'Here you go: (code block omitted) Use run_backtest() next.');
});

test('listenAndReply sends the recognized text to chat() and speaks the (stripped) reply', async () => {
  let sentMessage = '';
  const akanade = fakeAkanade((url, init) => {
    assert.ok(url.endsWith('/api/chat/message'));
    sentMessage = JSON.parse(init.body as string).message;
    return {
      status: 200,
      body: { reply: 'AAPL returned 12%.\n\n![chart](data:image/png;base64,abc)', pending_confirmation: false, confirm_prompt: null, provider: 'groq' },
    };
  });
  const { voice, spokenLines } = fakeVoice('Backtest AAPL for the last 10 years');
  const session = new VoiceChatSession({ akanade, sessionId: 's1', voice });

  const result = await session.listenAndReply();

  assert.equal(sentMessage, 'Backtest AAPL for the last 10 years');
  assert.equal(result.heard, 'Backtest AAPL for the last 10 years');
  assert.equal(result.spoken, 'AAPL returned 12%.');
  assert.deepEqual(spokenLines, ['AAPL returned 12%.']);
});

test('listenAndReply does not call speak() when nothing speakable remains after stripping', async () => {
  const akanade = fakeAkanade(() => ({
    status: 200,
    body: { reply: '![chart](data:image/png;base64,abc)', pending_confirmation: false, confirm_prompt: null, provider: 'groq' },
  }));
  const { voice, spokenLines } = fakeVoice('show me the chart');
  const session = new VoiceChatSession({ akanade, sessionId: 's1', voice });

  const result = await session.listenAndReply();

  assert.equal(result.spoken, '');
  assert.deepEqual(spokenLines, []);
});

test('listenAndConfirm recognizes an affirmative utterance and sends "yes" as the confirmation', async () => {
  let sentMessage = '';
  const akanade = fakeAkanade((url, init) => {
    sentMessage = JSON.parse(init.body as string).message;
    return { status: 200, body: { reply: 'Done.', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } };
  });
  const { voice } = fakeVoice('yes go ahead');
  const session = new VoiceChatSession({ akanade, sessionId: 's1', voice });

  await session.listenAndConfirm();

  assert.equal(sentMessage, 'yes');
});

test('listenAndConfirm sends "no" for a non-affirmative utterance', async () => {
  let sentMessage = '';
  const akanade = fakeAkanade((url, init) => {
    sentMessage = JSON.parse(init.body as string).message;
    return { status: 200, body: { reply: 'Cancelled.', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } };
  });
  const { voice } = fakeVoice('no, cancel that');
  const session = new VoiceChatSession({ akanade, sessionId: 's1', voice });

  await session.listenAndConfirm();

  assert.equal(sentMessage, 'no');
});
