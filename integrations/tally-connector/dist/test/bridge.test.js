"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const strict_1 = __importDefault(require("node:assert/strict"));
const node_test_1 = require("node:test");
const connector_1 = require("@algo-magneto/connector");
const bridge_1 = require("../src/bridge");
const SAMPLE_LEDGERS = [
    { name: 'Cash', parent: 'Cash-in-hand', openingBalance: 10000, closingBalance: 8500 },
    { name: 'ABC Traders', parent: 'Sundry Debtors', openingBalance: 0, closingBalance: -15000 },
];
const SAMPLE_VOUCHERS = [
    { date: '20240405', voucherType: 'Payment', voucherNumber: '12', partyName: 'ABC Traders', amount: -15000, narration: 'Office rent' },
];
function fakeTally(overrides = {}) {
    return {
        exportLedgers: async () => SAMPLE_LEDGERS,
        exportDayBook: async () => SAMPLE_VOUCHERS,
        ...overrides,
    };
}
function fakeMagneto(handler) {
    const fetchImpl = (async (input, init = {}) => {
        const { status, body } = handler(String(input), init);
        return { ok: status >= 200 && status < 300, status, statusText: String(status), text: async () => JSON.stringify(body) };
    });
    return new connector_1.MagnetoClient({ baseUrl: 'https://example.test', accessToken: 'tok', fetchImpl });
}
(0, node_test_1.test)('analyzeLedgers exports ledgers from Tally, embeds them in a chat prompt, and returns the reply', async () => {
    let sentPrompt = '';
    const magneto = fakeMagneto((url, init) => {
        strict_1.default.ok(url.endsWith('/api/chat/message'));
        sentPrompt = JSON.parse(init.body).message;
        return { status: 200, body: { reply: 'Cash balance looks healthy.', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } };
    });
    const bridge = new bridge_1.TallyMagnetoBridge({ magneto, tally: fakeTally() });
    const reply = await bridge.analyzeLedgers('My Company', 'Which ledgers have a negative closing balance?');
    strict_1.default.equal(reply, 'Cash balance looks healthy.');
    strict_1.default.match(sentPrompt, /My Company/);
    strict_1.default.match(sentPrompt, /ABC Traders/);
    strict_1.default.match(sentPrompt, /Which ledgers have a negative closing balance\?/);
});
(0, node_test_1.test)('analyzeDayBook exports vouchers for the date range and includes them in the prompt', async () => {
    let sentPrompt = '';
    let exportedFrom = '';
    let exportedTo = '';
    const magneto = fakeMagneto(() => ({
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
    const bridge = new bridge_1.TallyMagnetoBridge({ magneto, tally });
    const reply = await bridge.analyzeDayBook('My Company', '20240401', '20240430', 'Summarize this month.');
    strict_1.default.equal(exportedFrom, '20240401');
    strict_1.default.equal(exportedTo, '20240430');
    strict_1.default.equal(reply, 'One large rent payment on Apr 5.');
});
(0, node_test_1.test)('trainOnLedgers exports ledgers and uploads them via Train Algo AI, returning its result', async () => {
    const magneto = fakeMagneto((url, init) => {
        strict_1.default.ok(url.endsWith('/api/peft-training/upload'));
        strict_1.default.ok(init.body instanceof FormData);
        return {
            status: 200,
            body: { domain: 'trading', topic: 'Ledger export', source: 'Ledger export.txt', chars_logged: 80, rag_chunks_indexed: 1, job_status_note: null },
        };
    });
    const bridge = new bridge_1.TallyMagnetoBridge({ magneto, tally: fakeTally() });
    const result = await bridge.trainOnLedgers('My Company', 'trading', 'Ledger export');
    strict_1.default.equal(result.domain, 'trading');
    strict_1.default.equal(result.charsLogged, 80);
});
(0, node_test_1.test)('a Tally export failure propagates instead of silently returning an empty analysis', async () => {
    const magneto = fakeMagneto(() => {
        throw new Error('chat should not be called if the Tally export failed');
    });
    const tally = fakeTally({
        exportLedgers: async () => {
            throw new Error('Could not reach Tally');
        },
    });
    const bridge = new bridge_1.TallyMagnetoBridge({ magneto, tally });
    await strict_1.default.rejects(() => bridge.analyzeLedgers('My Company', 'anything'), /Could not reach Tally/);
});
