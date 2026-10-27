import assert from 'node:assert/strict';
import { test } from 'node:test';
import { AkanadeClient } from '@algo-akanade/connector';
import { TallyAkanadeBridge } from '../src/bridge';
import type { TallyGatewayLike, TallyLedger, TallyVoucher } from '../src/tallyGateway';

const SAMPLE_LEDGERS: TallyLedger[] = [
  { name: 'Cash', parent: 'Cash-in-hand', openingBalance: 10000, closingBalance: 8500 },
  { name: 'ABC Traders', parent: 'Sundry Debtors', openingBalance: 0, closingBalance: -15000 },
];

const SAMPLE_VOUCHERS: TallyVoucher[] = [
  { date: '20240405', voucherType: 'Payment', voucherNumber: '12', partyName: 'ABC Traders', amount: -15000, narration: 'Office rent' },
];

function fakeTally(overrides: Partial<TallyGatewayLike> = {}): TallyGatewayLike {
  return {
    exportLedgers: async () => SAMPLE_LEDGERS,
    exportDayBook: async () => SAMPLE_VOUCHERS,
    ...overrides,
  };
}

function fakeAkanade(handler: (url: string, init: RequestInit) => { status: number; body: unknown }): AkanadeClient {
  const fetchImpl = (async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const { status, body } = handler(String(input), init);
    return { ok: status >= 200 && status < 300, status, statusText: String(status), text: async () => JSON.stringify(body) } as Response;
  }) as typeof fetch;
  return new AkanadeClient({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });
}

test('analyzeLedgers exports ledgers from Tally, embeds them in a chat prompt, and returns the reply', async () => {
  let sentPrompt = '';
  const akanade = fakeAkanade((url, init) => {
    assert.ok(url.endsWith('/api/chat/message'));
    sentPrompt = JSON.parse(init.body as string).message;
    return { status: 200, body: { reply: 'Cash balance looks healthy.', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } };
  });
  const bridge = new TallyAkanadeBridge({ akanade, tally: fakeTally() });

  const reply = await bridge.analyzeLedgers('My Company', 'Which ledgers have a negative closing balance?');

  assert.equal(reply, 'Cash balance looks healthy.');
  assert.match(sentPrompt, /My Company/);
  assert.match(sentPrompt, /ABC Traders/);
  assert.match(sentPrompt, /Which ledgers have a negative closing balance\?/);
});

test('analyzeDayBook exports vouchers for the date range and includes them in the prompt', async () => {
  let sentPrompt = '';
  let exportedFrom = '';
  let exportedTo = '';
  const akanade = fakeAkanade(() => ({
    status: 200,
    body: { reply: 'One large rent payment on Apr 5.', pending_confirmation: false, confirm_prompt: null, provider: 'groq' },
  }));
  const tally = fakeTally({
    exportDayBook: async (_company, from, to) => {
      exportedFrom = from;
      exportedTo = to;
      return SAMPLE_VOUCHERS;
    },
  });
  const bridge = new TallyAkanadeBridge({ akanade, tally });

  const reply = await bridge.analyzeDayBook('My Company', '20240401', '20240430', 'Summarize this month.');

  assert.equal(exportedFrom, '20240401');
  assert.equal(exportedTo, '20240430');
  assert.equal(reply, 'One large rent payment on Apr 5.');
});

test('trainOnLedgers exports ledgers and uploads them via Train Algo AI, returning its result', async () => {
  const akanade = fakeAkanade((url, init) => {
    assert.ok(url.endsWith('/api/peft-training/upload'));
    assert.ok(init.body instanceof FormData);
    return {
      status: 200,
      body: { domain: 'trading', topic: 'Ledger export', source: 'Ledger export.txt', chars_logged: 80, rag_chunks_indexed: 1, job_status_note: null },
    };
  });
  const bridge = new TallyAkanadeBridge({ akanade, tally: fakeTally() });

  const result = await bridge.trainOnLedgers('My Company', 'trading', 'Ledger export');

  assert.equal(result.domain, 'trading');
  assert.equal(result.charsLogged, 80);
});

test('a Tally export failure propagates instead of silently returning an empty analysis', async () => {
  const akanade = fakeAkanade(() => {
    throw new Error('chat should not be called if the Tally export failed');
  });
  const tally = fakeTally({
    exportLedgers: async () => {
      throw new Error('Could not reach Tally');
    },
  });
  const bridge = new TallyAkanadeBridge({ akanade, tally });

  await assert.rejects(() => bridge.analyzeLedgers('My Company', 'anything'), /Could not reach Tally/);
});
