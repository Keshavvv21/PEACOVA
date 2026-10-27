// The actual "existing software to SLMs" glue: pulls real data out of
// Tally via tallyGateway.ts, then forwards it to ALGO AKANADE's existing
// /api/chat/message and /api/peft-training/* endpoints through
// @algo-akanade/connector — the same REST surface every ALGO AKANADE client
// (web, mobile, VS Code) already uses.
import type { AkanadeClient, TrainAlgoAiResult } from '@algo-akanade/connector';
import { TallyGatewayClient, type TallyGatewayLike, type TallyLedger, type TallyVoucher } from './tallyGateway';

export interface TallyAkanadeBridgeOptions {
  akanade: AkanadeClient;
  /** Defaults to a real TallyGatewayClient() talking to localhost:9000; override with a test double or a custom host/port. */
  tally?: TallyGatewayLike;
}

const MAX_ROWS_IN_PROMPT = 200; // keeps the chat prompt bounded — a real company can have thousands of ledgers/vouchers

function formatLedgersForPrompt(ledgers: TallyLedger[]): string {
  return ledgers
    .slice(0, MAX_ROWS_IN_PROMPT)
    .map((l) => `${l.name} | parent: ${l.parent ?? '-'} | opening: ${l.openingBalance ?? '-'} | closing: ${l.closingBalance ?? '-'}`)
    .join('\n');
}

function formatVouchersForPrompt(vouchers: TallyVoucher[]): string {
  return vouchers
    .slice(0, MAX_ROWS_IN_PROMPT)
    .map((v) => `${v.date ?? '-'} | ${v.voucherType ?? '-'} #${v.voucherNumber ?? '-'} | ${v.partyName ?? '-'} | ${v.amount ?? '-'} | ${v.narration ?? ''}`)
    .join('\n');
}

export class TallyAkanadeBridge {
  private akanade: AkanadeClient;
  private tally: TallyGatewayLike;

  constructor(options: TallyAkanadeBridgeOptions) {
    this.akanade = options.akanade;
    this.tally = options.tally ?? new TallyGatewayClient();
  }

  /** Exports ledger balances from Tally and asks ALGO AKANADE's chat pipeline (Trading Agent / PM Tool, whichever the question calls for) about them. */
  async analyzeLedgers(companyName: string, question: string, sessionId = 'tally-bridge'): Promise<string> {
    const ledgers = await this.tally.exportLedgers(companyName);
    const prompt =
      `Here are the ledger balances exported from Tally for "${companyName}":\n\n` +
      `name | parent | opening balance | closing balance\n${formatLedgersForPrompt(ledgers)}\n\n${question}`;
    const reply = await this.akanade.chat(prompt, sessionId);
    return reply.reply;
  }

  /** Exports the Day Book (vouchers) for a date range and asks a question about it. Dates must be Tally's own YYYYMMDD format. */
  async analyzeDayBook(companyName: string, fromDate: string, toDate: string, question: string, sessionId = 'tally-bridge'): Promise<string> {
    const vouchers = await this.tally.exportDayBook(companyName, fromDate, toDate);
    const prompt =
      `Here are the Day Book entries exported from Tally for "${companyName}" (${fromDate} to ${toDate}):\n\n` +
      `date | type | party | amount | narration\n${formatVouchersForPrompt(vouchers)}\n\n${question}`;
    const reply = await this.akanade.chat(prompt, sessionId);
    return reply.reply;
  }

  /** Feeds real, freshly-exported Tally ledger data into Train Algo AI as a ground-truth example for `domain`'s next nightly fine-tune. */
  async trainOnLedgers(companyName: string, domain: string, topic: string): Promise<TrainAlgoAiResult> {
    const ledgers = await this.tally.exportLedgers(companyName);
    const text = `Ledger balances for ${companyName}:\n${formatLedgersForPrompt(ledgers)}`;
    return this.akanade.trainAlgoAiText(domain, topic, text);
  }
}
