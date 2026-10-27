"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TallyMagnetoBridge = void 0;
const tallyGateway_1 = require("./tallyGateway");
const MAX_ROWS_IN_PROMPT = 200; // keeps the chat prompt bounded — a real company can have thousands of ledgers/vouchers
function formatLedgersForPrompt(ledgers) {
    return ledgers
        .slice(0, MAX_ROWS_IN_PROMPT)
        .map((l) => `${l.name} | parent: ${l.parent ?? '-'} | opening: ${l.openingBalance ?? '-'} | closing: ${l.closingBalance ?? '-'}`)
        .join('\n');
}
function formatVouchersForPrompt(vouchers) {
    return vouchers
        .slice(0, MAX_ROWS_IN_PROMPT)
        .map((v) => `${v.date ?? '-'} | ${v.voucherType ?? '-'} #${v.voucherNumber ?? '-'} | ${v.partyName ?? '-'} | ${v.amount ?? '-'} | ${v.narration ?? ''}`)
        .join('\n');
}
class TallyMagnetoBridge {
    constructor(options) {
        this.magneto = options.magneto;
        this.tally = options.tally ?? new tallyGateway_1.TallyGatewayClient();
    }
    /** Exports ledger balances from Tally and asks MAGNETO's chat pipeline (Trading Agent / PM Tool, whichever the question calls for) about them. */
    async analyzeLedgers(companyName, question, sessionId = 'tally-bridge') {
        const ledgers = await this.tally.exportLedgers(companyName);
        const prompt = `Here are the ledger balances exported from Tally for "${companyName}":\n\n` +
            `name | parent | opening balance | closing balance\n${formatLedgersForPrompt(ledgers)}\n\n${question}`;
        const reply = await this.magneto.chat(prompt, sessionId);
        return reply.reply;
    }
    /** Exports the Day Book (vouchers) for a date range and asks a question about it. Dates must be Tally's own YYYYMMDD format. */
    async analyzeDayBook(companyName, fromDate, toDate, question, sessionId = 'tally-bridge') {
        const vouchers = await this.tally.exportDayBook(companyName, fromDate, toDate);
        const prompt = `Here are the Day Book entries exported from Tally for "${companyName}" (${fromDate} to ${toDate}):\n\n` +
            `date | type | party | amount | narration\n${formatVouchersForPrompt(vouchers)}\n\n${question}`;
        const reply = await this.magneto.chat(prompt, sessionId);
        return reply.reply;
    }
    /** Feeds real, freshly-exported Tally ledger data into Train Algo AI as a ground-truth example for `domain`'s next nightly fine-tune. */
    async trainOnLedgers(companyName, domain, topic) {
        const ledgers = await this.tally.exportLedgers(companyName);
        const text = `Ledger balances for ${companyName}:\n${formatLedgersForPrompt(ledgers)}`;
        return this.magneto.trainAlgoAiText(domain, topic, text);
    }
}
exports.TallyMagnetoBridge = TallyMagnetoBridge;
