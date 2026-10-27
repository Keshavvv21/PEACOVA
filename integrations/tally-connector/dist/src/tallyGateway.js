"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TallyGatewayClient = exports.TallyGatewayError = void 0;
exports.escapeXml = escapeXml;
exports.buildLedgersRequest = buildLedgersRequest;
exports.buildDayBookRequest = buildDayBookRequest;
exports.parseLedgers = parseLedgers;
exports.parseDayBook = parseDayBook;
// Talks to Tally's own XML HTTP gateway — a real, documented Tally
// capability (Tally.ERP 9 / TallyPrime), enabled from inside Tally
// itself (Gateway of Tally -> F1 Help -> Settings -> Connectivity, or
// F12 -> Advanced Configuration depending on version) and listening on
// http://localhost:9000 by default. This module never talks to
// MAGNETO — see bridge.ts for the part that forwards exported data to
// MAGNETO's chat/Train-Algo-AI APIs.
//
// Honesty note: there is no real Tally installation in the environment
// this was built in, so the request/response shapes below are built
// from Tally's own published XML integration format, not verified
// end-to-end against a live instance. The parsers are deliberately
// tolerant (missing fields become null rather than throwing) because
// Tally's report-XML output has historically varied across versions —
// treat this as a solid starting point to verify against your own
// Tally install, not a guarantee.
const fast_xml_parser_1 = require("fast-xml-parser");
const DEFAULT_BASE_URL = 'http://localhost:9000';
class TallyGatewayError extends Error {
}
exports.TallyGatewayError = TallyGatewayError;
function escapeXml(value) {
    return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
/** Masters export — the most version-stable Tally XML request (a plain "List of Accounts" collection). */
function buildLedgersRequest(companyName) {
    return `<ENVELOPE>
 <HEADER>
  <TALLYREQUEST>Export Data</TALLYREQUEST>
 </HEADER>
 <BODY>
  <EXPORTDATA>
   <REQUESTDESC>
    <REPORTNAME>List of Accounts</REPORTNAME>
    <STATICVARIABLES>
     <SVCURRENTCOMPANY>${escapeXml(companyName)}</SVCURRENTCOMPANY>
    </STATICVARIABLES>
   </REQUESTDESC>
  </EXPORTDATA>
 </BODY>
</ENVELOPE>`;
}
/** Voucher export for a date range. `fromDate`/`toDate` must be Tally's own YYYYMMDD format (e.g. "20240401"). */
function buildDayBookRequest(companyName, fromDate, toDate) {
    return `<ENVELOPE>
 <HEADER>
  <TALLYREQUEST>Export Data</TALLYREQUEST>
 </HEADER>
 <BODY>
  <EXPORTDATA>
   <REQUESTDESC>
    <REPORTNAME>Day Book</REPORTNAME>
    <STATICVARIABLES>
     <SVFROMDATE>${escapeXml(fromDate)}</SVFROMDATE>
     <SVTODATE>${escapeXml(toDate)}</SVTODATE>
     <SVCURRENTCOMPANY>${escapeXml(companyName)}</SVCURRENTCOMPANY>
    </STATICVARIABLES>
   </REQUESTDESC>
  </EXPORTDATA>
 </BODY>
</ENVELOPE>`;
}
const xmlParser = new fast_xml_parser_1.XMLParser({ ignoreAttributes: false, attributeNamePrefix: '@_' });
function toArray(value) {
    if (value === undefined || value === null)
        return [];
    return Array.isArray(value) ? value : [value];
}
function toText(value) {
    if (value === undefined || value === null)
        return null;
    if (typeof value === 'object') {
        const obj = value;
        return obj['#text'] !== undefined ? String(obj['#text']) : null;
    }
    return String(value);
}
function toNumber(value) {
    const text = toText(value);
    if (text === null || text.trim() === '')
        return null;
    const n = Number(text.replace(/,/g, ''));
    return Number.isNaN(n) ? null : n;
}
/** Parses a "List of Accounts" (ledger masters) response into plain objects. Fields Tally omits become null, never thrown. */
function parseLedgers(xml) {
    const doc = xmlParser.parse(xml);
    const envelope = (doc.ENVELOPE ?? doc);
    const ledgers = toArray(envelope.LEDGER);
    return ledgers
        .map((l) => ({
        name: toText(l['@_NAME']) ?? toText(l.NAME) ?? '',
        parent: toText(l.PARENT),
        openingBalance: toNumber(l.OPENINGBALANCE),
        closingBalance: toNumber(l.CLOSINGBALANCE),
    }))
        .filter((l) => l.name !== '');
}
/** Parses a Day Book (voucher) response into plain objects. */
function parseDayBook(xml) {
    const doc = xmlParser.parse(xml);
    const envelope = (doc.ENVELOPE ?? doc);
    const vouchers = toArray(envelope.VOUCHER);
    return vouchers.map((v) => ({
        date: toText(v.DATE),
        voucherType: toText(v.VOUCHERTYPENAME),
        voucherNumber: toText(v.VOUCHERNUMBER),
        partyName: toText(v.PARTYLEDGERNAME) ?? toText(v.PARTYNAME),
        amount: toNumber(v.AMOUNT),
        narration: toText(v.NARRATION),
    }));
}
class TallyGatewayClient {
    constructor(options = {}) {
        this.baseUrl = (options.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, '');
        const impl = options.fetchImpl ?? globalThis.fetch;
        if (!impl)
            throw new Error('No fetch implementation available on this runtime — pass fetchImpl explicitly.');
        this.fetchImpl = impl;
    }
    async exportLedgers(companyName) {
        return parseLedgers(await this.postXml(buildLedgersRequest(companyName)));
    }
    async exportDayBook(companyName, fromDate, toDate) {
        return parseDayBook(await this.postXml(buildDayBookRequest(companyName, fromDate, toDate)));
    }
    async postXml(requestXml) {
        let res;
        try {
            res = await this.fetchImpl(this.baseUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'text/xml' },
                body: requestXml,
            });
        }
        catch {
            throw new TallyGatewayError(`Could not reach Tally's HTTP gateway at ${this.baseUrl}. Confirm Tally is running on this ` +
                `machine and its HTTP/ODBC server is enabled (Gateway of Tally -> F1 Help -> Settings -> ` +
                `Connectivity, or F12 -> Advanced Configuration, depending on your Tally version) on this port.`);
        }
        const text = await res.text();
        if (!res.ok) {
            throw new TallyGatewayError(`Tally gateway returned HTTP ${res.status}: ${text.slice(0, 500)}`);
        }
        if (/<LINEERROR>/i.test(text)) {
            throw new TallyGatewayError(`Tally reported an error for this request: ${text.slice(0, 500)}`);
        }
        return text;
    }
}
exports.TallyGatewayClient = TallyGatewayClient;
