import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  buildDayBookRequest,
  buildLedgersRequest,
  parseDayBook,
  parseLedgers,
  TallyGatewayClient,
  TallyGatewayError,
} from '../src/tallyGateway';

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

test('buildLedgersRequest embeds the company name and escapes XML-special characters', () => {
  const xml = buildLedgersRequest('R&D Traders <Pvt Ltd>');
  assert.match(xml, /<TALLYREQUEST>Export Data<\/TALLYREQUEST>/);
  assert.match(xml, /<REPORTNAME>List of Accounts<\/REPORTNAME>/);
  assert.match(xml, /<SVCURRENTCOMPANY>R&amp;D Traders &lt;Pvt Ltd&gt;<\/SVCURRENTCOMPANY>/);
});

test('buildDayBookRequest embeds the date range in Tally-native YYYYMMDD form', () => {
  const xml = buildDayBookRequest('My Company', '20240401', '20240630');
  assert.match(xml, /<REPORTNAME>Day Book<\/REPORTNAME>/);
  assert.match(xml, /<SVFROMDATE>20240401<\/SVFROMDATE>/);
  assert.match(xml, /<SVTODATE>20240630<\/SVTODATE>/);
});

test('parseLedgers extracts name (from the NAME attribute), parent, and both balances', () => {
  const ledgers = parseLedgers(SAMPLE_LEDGERS_XML);
  assert.equal(ledgers.length, 2);
  assert.deepEqual(ledgers[0], { name: 'Cash', parent: 'Cash-in-hand', openingBalance: 10000, closingBalance: 8500 });
  assert.deepEqual(ledgers[1], {
    name: 'ABC Traders',
    parent: 'Sundry Debtors',
    openingBalance: 0,
    closingBalance: -15000,
  });
});

test('parseLedgers tolerates missing fields instead of throwing', () => {
  const ledgers = parseLedgers('<ENVELOPE><LEDGER NAME="Bare Ledger"></LEDGER></ENVELOPE>');
  assert.deepEqual(ledgers, [{ name: 'Bare Ledger', parent: null, openingBalance: null, closingBalance: null }]);
});

test('parseLedgers returns an empty array for a response with no LEDGER nodes', () => {
  assert.deepEqual(parseLedgers('<ENVELOPE></ENVELOPE>'), []);
});

test('parseDayBook extracts every voucher field, coercing AMOUNT to a signed number', () => {
  const vouchers = parseDayBook(SAMPLE_DAYBOOK_XML);
  assert.equal(vouchers.length, 2);
  assert.deepEqual(vouchers[0], {
    date: '20240405',
    voucherType: 'Payment',
    voucherNumber: '12',
    partyName: 'ABC Traders',
    amount: -15000,
    narration: 'Office rent for April',
  });
  assert.equal(vouchers[1].amount, 25000);
  assert.equal(vouchers[1].partyName, 'XYZ Corp');
});

test('a single VOUCHER (not an array) still parses as a one-element list', () => {
  const singleVoucherXml = `<ENVELOPE><VOUCHER><DATE>20240101</DATE><VOUCHERTYPENAME>Sales</VOUCHERTYPENAME></VOUCHER></ENVELOPE>`;
  const vouchers = parseDayBook(singleVoucherXml);
  assert.equal(vouchers.length, 1);
  assert.equal(vouchers[0].voucherType, 'Sales');
});

test('TallyGatewayClient.exportLedgers posts XML to the gateway and returns parsed ledgers', async () => {
  let capturedUrl = '';
  let capturedBody = '';
  const fetchImpl = (async (input: RequestInfo | URL, init: RequestInit = {}) => {
    capturedUrl = String(input);
    capturedBody = String(init.body);
    return { ok: true, status: 200, text: async () => SAMPLE_LEDGERS_XML } as Response;
  }) as typeof fetch;

  const client = new TallyGatewayClient({ baseUrl: 'http://localhost:9000', fetchImpl });
  const ledgers = await client.exportLedgers('My Company');

  assert.equal(capturedUrl, 'http://localhost:9000');
  assert.match(capturedBody, /<SVCURRENTCOMPANY>My Company<\/SVCURRENTCOMPANY>/);
  assert.equal(ledgers.length, 2);
});

test('a network failure reaching Tally raises a helpful TallyGatewayError, not a raw fetch error', async () => {
  const fetchImpl = (async () => {
    throw new Error('ECONNREFUSED');
  }) as unknown as typeof fetch;
  const client = new TallyGatewayClient({ baseUrl: 'http://localhost:9000', fetchImpl });
  await assert.rejects(
    () => client.exportLedgers('My Company'),
    (err: unknown) => err instanceof TallyGatewayError && /Could not reach Tally/.test(err.message),
  );
});

test('a <LINEERROR> in an otherwise-200 response is surfaced as a TallyGatewayError', async () => {
  const fetchImpl = (async () =>
    ({ ok: true, status: 200, text: async () => '<ENVELOPE><LINEERROR>Company not found</LINEERROR></ENVELOPE>' }) as Response) as typeof fetch;
  const client = new TallyGatewayClient({ baseUrl: 'http://localhost:9000', fetchImpl });
  await assert.rejects(
    () => client.exportLedgers('Nonexistent Co'),
    (err: unknown) => err instanceof TallyGatewayError && /Company not found/.test(err.message),
  );
});
