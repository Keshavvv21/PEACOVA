# @algo-akanade/connector

A small, plain client for ALGO AKANADE's existing REST API — not a
standalone "SDK product," just the handful of functions any external
caller needs around the endpoints already documented in
[`README_EXTERNAL_API.md`](../../orchestrator_framework_layer/akanade/README_EXTERNAL_API.md).
No new backend endpoints were built for this package; it wraps what
already ships to the web app, mobile app, and VS Code extension.

Used internally by [`../tally-connector`](../tally-connector), and
directly importable by any React or React Native app that wants to
talk to ALGO AKANADE without hand-rolling `fetch` calls.

## Install

This isn't published to a registry. Use it as a local dependency:

```json
{ "dependencies": { "@algo-akanade/connector": "file:../algo-akanade-connector" } }
```

or copy `dist/src` into your own project.

```bash
npm install
npm run build
```

## Usage

```ts
import { AkanadeClient } from '@algo-akanade/connector';

const client = new AkanadeClient({ baseUrl: 'https://your-host/chatbot' });
await client.login('you@company.com', 'your-password');

const reply = await client.chat('Backtest AAPL for the last 10 years', 'session-1');
console.log(reply.reply); // includes CAGR/volatility/Sharpe and an embedded chart image

// Feed real data back in as a training example (Train Algo AI):
await client.trainAlgoAiText('trading', 'Q3 ledger summary', someExtractedText);

// Admin token only — silently returns null for anyone else instead of throwing:
const jobs = await client.runningTrainingJobs();
```

Works in Node 18+, browsers, and React Native as-is — all three have a
global `fetch`/`FormData`/`Blob`. Pass `fetchImpl` in the constructor
for any runtime that doesn't.

## What this does NOT do

- No offline queueing/retry — a caller integrating with flaky hardware
  (a POS terminal, an on-prem ERP box) should wrap calls with its own
  retry policy.
- No structured (non-chat) access to Trading Agent / PM Tool data —
  `chat()` returns natural-language replies with embedded data (charts,
  numbers in prose), because that is the real, current shape of
  `/api/chat/message`. There is no separate structured
  `/api/trading/backtest` endpoint today.
- No token refresh — catch a 401 (`AkanadeApiError` with `status===401`)
  and call `login()` again.

## Tests

```bash
npm test
```

7 tests, all against a mocked `fetch` (no real backend needed to run
them) — see `test/client.test.ts`.
