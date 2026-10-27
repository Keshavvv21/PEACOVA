export interface TallyGatewayOptions {
    /** Default: http://localhost:9000 — Tally's own default HTTP gateway port. */
    baseUrl?: string;
    fetchImpl?: typeof fetch;
}
export interface TallyLedger {
    name: string;
    parent: string | null;
    openingBalance: number | null;
    closingBalance: number | null;
}
export interface TallyVoucher {
    /** Tally's own YYYYMMDD date string, left unconverted so callers pick their own format. */
    date: string | null;
    voucherType: string | null;
    voucherNumber: string | null;
    partyName: string | null;
    amount: number | null;
    narration: string | null;
}
export interface TallyGatewayLike {
    exportLedgers(companyName: string): Promise<TallyLedger[]>;
    exportDayBook(companyName: string, fromDate: string, toDate: string): Promise<TallyVoucher[]>;
}
export declare class TallyGatewayError extends Error {
}
export declare function escapeXml(value: string): string;
/** Masters export — the most version-stable Tally XML request (a plain "List of Accounts" collection). */
export declare function buildLedgersRequest(companyName: string): string;
/** Voucher export for a date range. `fromDate`/`toDate` must be Tally's own YYYYMMDD format (e.g. "20240401"). */
export declare function buildDayBookRequest(companyName: string, fromDate: string, toDate: string): string;
/** Parses a "List of Accounts" (ledger masters) response into plain objects. Fields Tally omits become null, never thrown. */
export declare function parseLedgers(xml: string): TallyLedger[];
/** Parses a Day Book (voucher) response into plain objects. */
export declare function parseDayBook(xml: string): TallyVoucher[];
export declare class TallyGatewayClient implements TallyGatewayLike {
    private baseUrl;
    private fetchImpl;
    constructor(options?: TallyGatewayOptions);
    exportLedgers(companyName: string): Promise<TallyLedger[]>;
    exportDayBook(companyName: string, fromDate: string, toDate: string): Promise<TallyVoucher[]>;
    private postXml;
}
