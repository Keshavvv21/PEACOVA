"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const strict_1 = __importDefault(require("node:assert/strict"));
const node_test_1 = require("node:test");
const tallyGateway_1 = require("../src/tallyGateway");
// Realistic sample response shapes, built from Tally's own published
// XML integration format (Export Data / List of Accounts / Day Book).
// Not captured from a live Tally instance — see the README's honesty
// note — but these are the field names Tally's documentation and
// integration guides consistently use.
const SAMPLE_LEDGERS_XML = `<ENVELOPE>
  <LEDGER NAME="Cash">
    <PARENT>Cash-in-hand</PARENT>
    <OPENINGBALANCE>10000.00</OPENINGBALANCE>
    <CLOSINGBALANCE>8500.00</CLOSINGBALANCE>
  </LEDGER>
  <LEDGER NAME="ABC Traders">
    <PARENT>Sundry Debtors</PARENT>
    <OPENINGBALANCE>0.00</OPENINGBALANCE>
    <CLOSINGBALANCE>-15000.00</CLOSINGBALANCE>
  </LEDGER>
</ENVELOPE>`;
const SAMPLE_DAYBOOK_XML = `<ENVELOPE>
  <VOUCHER>
    <DATE>20240405</DATE>
    <VOUCHERTYPENAME>Payment</VOUCHERTYPENAME>
    <VOUCHERNUMBER>12</VOUCHERNUMBER>
    <PARTYLEDGERNAME>ABC Traders</PARTYLEDGERNAME>
    <AMOUNT>-15000.00</AMOUNT>
    <NARRATION>Office rent for April</NARRATION>
  </VOUCHER>
  <VOUCHER>
    <DATE>20240410</DATE>
    <VOUCHERTYPENAME>Receipt</VOUCHERTYPENAME>
    <VOUCHERNUMBER>7</VOUCHERNUMBER>
    <PARTYLEDGERNAME>XYZ Corp</PARTYLEDGERNAME>
    <AMOUNT>25000.00</AMOUNT>
    <NARRATION>Advance received</NARRATION>
  </VOUCHER>
</ENVELOPE>`;
(0, node_test_1.test)('buildLedgersRequest embeds the company name and escapes XML-special characters', () => {
    const xml = (0, tallyGateway_1.buildLedgersRequest)('R&D Traders <Pvt Ltd>');
    strict_1.default.match(xml, /<TALLYREQUEST>Export Data<\/TALLYREQUEST>/);
    strict_1.default.match(xml, /<REPORTNAME>List of Accounts<\/REPORTNAME>/);
    strict_1.default.match(xml, /<SVCURRENTCOMPANY>R&amp;D Traders &lt;Pvt Ltd&gt;<\/SVCURRENTCOMPANY>/);
});
(0, node_test_1.test)('buildDayBookRequest embeds the date range in Tally-native YYYYMMDD form', () => {
    const xml = (0, tallyGateway_1.buildDayBookRequest)('My Company', '20240401', '20240630');
    strict_1.default.match(xml, /<REPORTNAME>Day Book<\/REPORTNAME>/);
    strict_1.default.match(xml, /<SVFROMDATE>20240401<\/SVFROMDATE>/);
    strict_1.default.match(xml, /<SVTODATE>20240630<\/SVTODATE>/);
});
(0, node_test_1.test)('parseLedgers extracts name (from the NAME attribute), parent, and both balances', () => {
    const ledgers = (0, tallyGateway_1.parseLedgers)(SAMPLE_LEDGERS_XML);
    strict_1.default.equal(ledgers.length, 2);
    strict_1.default.deepEqual(ledgers[0], { name: 'Cash', parent: 'Cash-in-hand', openingBalance: 10000, closingBalance: 8500 });
    strict_1.default.deepEqual(ledgers[1], {
        name: 'ABC Traders',
        parent: 'Sundry Debtors',
        openingBalance: 0,
        closingBalance: -15000,
    });
});
(0, node_test_1.test)('parseLedgers tolerates missing fields instead of throwing', () => {
    const ledgers = (0, tallyGateway_1.parseLedgers)('<ENVELOPE><LEDGER NAME="Bare Ledger"></LEDGER></ENVELOPE>');
    strict_1.default.deepEqual(ledgers, [{ name: 'Bare Ledger', parent: null, openingBalance: null, closingBalance: null }]);
});
(0, node_test_1.test)('parseLedgers returns an empty array for a response with no LEDGER nodes', () => {
    strict_1.default.deepEqual((0, tallyGateway_1.parseLedgers)('<ENVELOPE></ENVELOPE>'), []);
});
(0, node_test_1.test)('parseDayBook extracts every voucher field, coercing AMOUNT to a signed number', () => {
    const vouchers = (0, tallyGateway_1.parseDayBook)(SAMPLE_DAYBOOK_XML);
    strict_1.default.equal(vouchers.length, 2);
    strict_1.default.deepEqual(vouchers[0], {
        date: '20240405',
        voucherType: 'Payment',
        voucherNumber: '12',
        partyName: 'ABC Traders',
        amount: -15000,
        narration: 'Office rent for April',
    });
    strict_1.default.equal(vouchers[1].amount, 25000);
    strict_1.default.equal(vouchers[1].partyName, 'XYZ Corp');
});
(0, node_test_1.test)('a single VOUCHER (not an array) still parses as a one-element list', () => {
    const singleVoucherXml = `<ENVELOPE><VOUCHER><DATE>20240101</DATE><VOUCHERTYPENAME>Sales</VOUCHERTYPENAME></VOUCHER></ENVELOPE>`;
    const vouchers = (0, tallyGateway_1.parseDayBook)(singleVoucherXml);
    strict_1.default.equal(vouchers.length, 1);
    strict_1.default.equal(vouchers[0].voucherType, 'Sales');
});
(0, node_test_1.test)('TallyGatewayClient.exportLedgers posts XML to the gateway and returns parsed ledgers', async () => {
    let capturedUrl = '';
    let capturedBody = '';
    const fetchImpl = (async (input, init = {}) => {
        capturedUrl = String(input);
        capturedBody = String(init.body);
        return { ok: true, status: 200, text: async () => SAMPLE_LEDGERS_XML };
    });
    const client = new tallyGateway_1.TallyGatewayClient({ baseUrl: 'http://localhost:9000', fetchImpl });
    const ledgers = await client.exportLedgers('My Company');
    strict_1.default.equal(capturedUrl, 'http://localhost:9000');
    strict_1.default.match(capturedBody, /<SVCURRENTCOMPANY>My Company<\/SVCURRENTCOMPANY>/);
    strict_1.default.equal(ledgers.length, 2);
});
(0, node_test_1.test)('a network failure reaching Tally raises a helpful TallyGatewayError, not a raw fetch error', async () => {
    const fetchImpl = (async () => {
        throw new Error('ECONNREFUSED');
    });
    const client = new tallyGateway_1.TallyGatewayClient({ baseUrl: 'http://localhost:9000', fetchImpl });
    await strict_1.default.rejects(() => client.exportLedgers('My Company'), (err) => err instanceof tallyGateway_1.TallyGatewayError && /Could not reach Tally/.test(err.message));
});
(0, node_test_1.test)('a <LINEERROR> in an otherwise-200 response is surfaced as a TallyGatewayError', async () => {
    const fetchImpl = (async () => ({ ok: true, status: 200, text: async () => '<ENVELOPE><LINEERROR>Company not found</LINEERROR></ENVELOPE>' }));
    const client = new tallyGateway_1.TallyGatewayClient({ baseUrl: 'http://localhost:9000', fetchImpl });
    await strict_1.default.rejects(() => client.exportLedgers('Nonexistent Co'), (err) => err instanceof tallyGateway_1.TallyGatewayError && /Company not found/.test(err.message));
});
