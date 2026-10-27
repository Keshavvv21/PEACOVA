# algo-tally-connector

A working Tally integration for ALGO AKANADE — pulls real ledger/voucher
data out of Tally through its own XML HTTP gateway, and forwards it to
ALGO AKANADE's chat and Train Algo AI APIs via
[`@algo-akanade/connector`](../algo-akanade-connector). This is a
concrete plugin for Tally, not a generic "SDK" — the same pattern
(export data → call ALGO AKANADE's existing REST API) generalizes to other
ERP/React/React Native apps by writing a different `*Gateway.ts`
against whatever that software's own export API is, and reusing the
bridge/server/connector as-is.

**Already have Tally installed and you're not a software person?** See
[`CONNECT_YOUR_TALLY.md`](CONNECT_YOUR_TALLY.md) — download one
program, double-click it, fill in one form in your browser, done. No
terminal, no npm, no TDL editing required (a TDL menu button is
optional, covered there too).

## How it fits together

```
Tally (desktop app)                    algo-tally-connector                   ALGO AKANADE backend
┌──────────────────┐   HTTP Post (TDL)  ┌──────────────────────┐  HTTPS/JSON  ┌──────────────────┐
│  Custom menu item │ ─────────────────▶│  Bridge server        │─────────────▶│ /api/chat/message │
│  (see ../tdl/)    │                    │  (src/server.ts)      │              │ /api/peft-training│
└──────────────────┘                    │   ↓ src/bridge.ts     │              └──────────────────┘
                                          │   ↓ src/tallyGateway.ts│
                                          └───────────┬───────────┘
                                                       │ XML over HTTP
                                                       ▼
                                          Tally's own gateway, port 9000
```

- **`src/tallyGateway.ts`** — builds Tally's documented "Export Data"
  XML requests (`List of Accounts` for ledger masters, `Day Book` for
  vouchers in a date range) and parses the XML response into plain
  objects. Talks only to Tally; knows nothing about ALGO AKANADE.
- **`src/bridge.ts`** — the actual glue: exports data from Tally, folds
  it into a chat prompt, and calls
  `AkanadeClient.chat()` / `.trainAlgoAiText()`.
- **`src/config.ts`** — reads/writes a plain JSON config file
  (`algo-tally-config.json`, next to wherever the program runs from) so
  a non-technical user never needs a terminal, environment variables,
  or a config file editor after the very first run.
- **`src/setupPage.ts`** / **`src/chatPage.ts`** — the two screens a
  user actually sees: a one-time setup form (`GET`/`POST /setup`), and
  the popup chat window itself (`GET /`) — plain HTML and vanilla JS,
  no build step, no framework, so the packaged executable stays a
  single file with zero runtime dependencies to install.
- **`src/server.ts`** / **`src/cli.ts`** — the local HTTP server tying
  it together (`GET /`, `GET`/`POST /setup`, `POST /chat`,
  `POST /analyze-ledgers`, `POST /analyze-daybook`,
  `POST /train-on-ledgers`, `GET /health`), plus `src/openBrowser.ts`
  which opens the user's default browser to the right screen on
  startup — so "run the program" is the entire installation experience.
- **`tdl/AlgoAkanadeAI.tdl`** — an *optional* TDL snippet adding an
  "Ask Algo AI" menu item to Tally's own Gateway, using TDL's built-in
  `HTTP Post` function (available since Tally.ERP 9 Series 6.x) to call
  the same server above. Not required — the popup chat window works on
  its own.

## Running it

**Not a software person?** See [`CONNECT_YOUR_TALLY.md`](CONNECT_YOUR_TALLY.md)
instead — download-and-double-click, no terminal.

For development, or to build the packaged executables yourself:

```bash
npm install
npm run build
npm start
```

This starts the same program `CONNECT_YOUR_TALLY.md` describes: it
opens a browser to `http://localhost:4600/setup` (first run) or
`http://localhost:4600/` (once configured), and saves whatever you
enter to `algo-tally-config.json` next to where it's running.

Skip the setup form during development by setting environment
variables before the very first run (seeds the config file once, then
behaves identically to the form):

```bash
ALGO_AKANADE_BASE_URL=https://your-host/chatbot \
ALGO_AKANADE_EMAIL=you@company.com \
ALGO_AKANADE_PASSWORD=your-password \
TALLY_BASE_URL=http://localhost:9000 \
npm start
```

Build the standalone, dependency-free executables (no Node.js install
required by whoever runs them):

```bash
npm run package:all
# -> dist-exe/AlgoAIForTally-windows.exe
# -> dist-exe/AlgoAIForTally-linux
```

Test any route directly, without opening a browser:

```bash
curl -X POST http://localhost:4600/analyze-ledgers \
  -H "Content-Type: application/json" \
  -d '{"companyName": "My Company", "question": "Which ledgers look overdue?"}'
```

## Honesty note — what has and hasn't been verified

There is no real Tally installation available in the environment this
was built in. What's verified:
- The XML request builders and parsers are tested against realistic
  sample XML built from Tally's own published integration format
  (`test/tallyGateway.test.ts`, 14 tests, all passing).
- The bridge and local HTTP server — including the full setup-form →
  saved-config → chat round-trip — are tested end-to-end against a
  mocked Tally gateway and a real (not mocked) fake HTTP ALGO AKANADE
  backend (`test/bridge.test.ts`, `test/server.test.ts`, `test/config.test.ts`,
  27 tests total).
- **The packaged Linux executable was actually built and run** in this
  session: `npm run package:linux`, then the resulting single-file
  binary was launched standalone, driven through `/setup` with a real
  HTTP request against a throwaway fake ALGO AKANADE server, confirmed to
  write its config file next to itself, and confirmed to complete a
  real `POST /chat` login+reply round-trip — not just unit tests.
- **The Windows executable was built** (`npm run package:win`) and
  confirmed to be a valid Windows binary (`file` reports `PE32+
  executable ... for MS Windows`) from the exact same source as the
  Linux binary that was run end-to-end, but could not itself be *run*
  in this sandbox (no Windows or Wine available).

What's **not** verified: an actual live Tally instance's exact XML
output. Tally's report-XML shape has varied across versions in
practice, which is why the parsers here are deliberately tolerant
(missing fields become `null`, never an exception) rather than a rigid
schema. Treat this as a solid, tested starting point to validate
against your own Tally install — not a guarantee it matches byte-for-byte
on every Tally version.

## Extending this to other software (React, React Native, other ERPs)

`src/tallyGateway.ts` is the only Tally-specific file. To connect a
different system:
1. Write your own `<system>Gateway.ts` that fetches/exports data from
   that system's own API (a React/React Native app can usually just
   call `@algo-akanade/connector` directly — see its own README — with
   no bridge server needed at all, since it already runs its own HTTP
   client).
2. Reuse `bridge.ts`'s pattern (export → format a prompt → `chat()`) or
   write your own — the ALGO AKANADE side of the contract never changes.

## Tests

```bash
npm test
```
27 tests, all passing, no live Tally or ALGO AKANADE instance required.
