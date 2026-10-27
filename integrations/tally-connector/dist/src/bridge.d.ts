import type { MagnetoClient, TrainAlgoAiResult } from '@algo-magneto/connector';
import { type TallyGatewayLike } from './tallyGateway';
export interface TallyMagnetoBridgeOptions {
    magneto: MagnetoClient;
    /** Defaults to a real TallyGatewayClient() talking to localhost:9000; override with a test double or a custom host/port. */
    tally?: TallyGatewayLike;
}
export declare class TallyMagnetoBridge {
    private magneto;
    private tally;
    constructor(options: TallyMagnetoBridgeOptions);
    /** Exports ledger balances from Tally and asks MAGNETO's chat pipeline (Trading Agent / PM Tool, whichever the question calls for) about them. */
    analyzeLedgers(companyName: string, question: string, sessionId?: string): Promise<string>;
    /** Exports the Day Book (vouchers) for a date range and asks a question about it. Dates must be Tally's own YYYYMMDD format. */
    analyzeDayBook(companyName: string, fromDate: string, toDate: string, question: string, sessionId?: string): Promise<string>;
    /** Feeds real, freshly-exported Tally ledger data into Train Algo AI as a ground-truth example for `domain`'s next nightly fine-tune. */
    trainOnLedgers(companyName: string, domain: string, topic: string): Promise<TrainAlgoAiResult>;
}
