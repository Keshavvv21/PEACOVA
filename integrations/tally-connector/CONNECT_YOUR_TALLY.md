# Connecting an already-installed Tally

You already have Tally (Tally.ERP 9 or TallyPrime) running with real
company data, and you're not a software person. This is the plain
version: download one file, run it, fill in one form, done. No
terminal, no npm, no code.

## What you'll end up with

A small program that sits quietly in the background and opens a chat
window — type or speak a question, get a real answer from Algo AI
about your Tally data. There's also an optional menu button inside
Tally itself for the same thing, if you want it (Step 5).

## Step 1 — Turn on Tally's own connection feature

This is a setting Tally already has — you're just switching it on,
not installing anything new inside Tally.

- **TallyPrime**: Gateway of Tally → **F1** (Help) → **Settings** →
  **Connectivity** → under "Client/Server configuration for Multi-User",
  set **TallyPrime acts as**: `Both` (or `Server`). Leave the **Port**
  as `9000`.
- **Tally.ERP 9**: Gateway of Tally → **F12** (Configure) → **Advanced
  Configuration** → set **Enable ODBC?**: `Yes`. Leave the **Port** as
  `9000`.

If Tally was already set up this way for something else (an Excel
add-on, another integration), skip this step — it's already on.

**After changing this, close and reopen Tally once** so the change
takes effect.

## Step 2 — Download and run the program

1. Get the program for your computer:
   - Windows: `AlgoAIForTally-windows.exe`
   - Ubuntu/Linux: `AlgoAIForTally-linux`
   (Ask whoever gave you this repository/link for these two files —
   they're built from `integrations/tally-connector` via `npm run
   package:all` and aren't stored in the source code itself.)
2. Put it somewhere you'll remember (e.g. your Desktop or Documents).
3. **Double-click it.**
   - Windows may show a "Windows protected your PC" warning the first
     time (this is Windows being cautious about any new program, not a
     sign something is wrong) — click **More info** → **Run anyway**.
   - Ubuntu may need you to right-click the file → **Properties** →
     **Permissions** → check **Allow executing file as program** once,
     the first time only.
4. A window titled something like "Algo AI for Tally is running" stays
   open in the background — **leave it open**; closing it turns Algo
   AI off. Your web browser should open on its own to a setup page. If
   it doesn't, open your browser and go to `http://localhost:4600/setup`.

## Step 3 — Fill in the setup form (one time only)

You'll see four fields:

- **Your Algo AI web address** — the same web address you already use
  to open Algo AI/ALGO AKANADE in a browser.
- **Your email** and **Your password** — your normal Algo AI login.
- **Tally's connection address** — leave this as `http://localhost:9000`
  unless someone told you Tally uses a different port.

Click **Save and start**. You'll see a "Saved" message with a link to
open the chat — click it.

## Step 4 — Use it

Type a question and press **Send**, or click the microphone button and
speak (the mic button only appears if your browser supports it — Chrome
and Edge do). Two buttons above the message box, **Analyze my
ledgers** and **Analyze this month's Day Book**, ask Algo AI about
your real Tally data directly — you'll be asked for your company name
exactly as it appears in Tally, and (for Day Book) a date range.

If a question needs a yes/no confirmation, two buttons appear for
that — no need to type "yes" or "no".

**To change your login or address later**, click **Settings** at the
top of the chat window — it reopens the same form from Step 3.

## Step 5 — (Optional) Add a menu button inside Tally itself

If you'd like an "Ask Algo AI" entry on Tally's own Gateway menu, in
addition to the popup chat window:

Gateway of Tally → **F12** → **TDL Configuration** → add the file
[`tdl/AlgoAkanadeAI.tdl`](tdl/AlgoAkanadeAI.tdl) as a local TDL file →
restart Tally. This is entirely optional — the popup chat window from
Step 4 already does everything on its own.

## If something doesn't work

- **The setup page never opens in my browser**: open your browser
  yourself and go to `http://localhost:4600/setup`.
- **"Could not reach Tally"** when using a ledger/Day Book question:
  Tally needs to be open with the company loaded, and Step 1's setting
  needs Tally to have been restarted after you turned it on.
- **Empty or wrong answers about ledgers**: the company name you typed
  needs to match Tally's own name for it **exactly**, including
  capitalization — check the top-left of the Gateway of Tally screen.
- **The program won't start / antivirus blocks it**: some antivirus
  software is cautious about any program packaged the way this one is
  (a single downloaded .exe/.exe-like file); add an exception for it
  the same way you would for any other trusted downloaded tool, or ask
  whoever manages your computer to do this.
- **I want to stop it from running**: close the window from Step 2. To
  remove it completely, delete the program file and the small
  `algo-tally-config.json` file that appears next to it (that's the
  only thing it creates on your computer).

## What this does and doesn't touch in your existing Tally

- **Read-only.** Every request this makes is a read ("Export Data")
  request — it never writes vouchers, edits ledgers, or changes any
  Tally setting beyond the one-time connection toggle in Step 1, which
  you likely already use for something else if your business does any
  Tally integration today.
- **No new Tally company, no data migration.** It reads whatever
  company is already open in Tally, live, exactly as your accountants
  see it.
- Your email/password are saved in one plain file
  (`algo-tally-config.json`) next to the program, on this computer
  only — never sent anywhere except the Algo AI address you entered.

---

## For a software person: running from source instead of the .exe

If you'd rather run this from source (development, or building the
`.exe`/binary yourself), see [`README.md`](README.md) — the short
version:

```bash
cd integrations/tally-connector
npm install
npm run build
npm start          # opens the same setup page as Step 2/3 above
```

Or set environment variables once to skip the setup form (useful for
CI or a scripted install):

```bash
ALGO_AKANADE_BASE_URL=https://your-host/chatbot \
ALGO_AKANADE_EMAIL=you@company.com \
ALGO_AKANADE_PASSWORD=your-password \
TALLY_BASE_URL=http://localhost:9000 \
npm start
```

To build the two downloadable programs from Step 2 yourself:

```bash
npm run package:all
# produces dist-exe/AlgoAIForTally-windows.exe and dist-exe/AlgoAIForTally-linux
```

Verified end-to-end in the environment this was built in: the Linux
binary was built, run, taken through the setup form against a fake
ALGO AKANADE backend, and confirmed to save its config file and answer a
real chat round-trip — not just unit tests. The Windows binary was
built and confirmed to be a valid Windows executable (`file` reports
`PE32+ executable ... for MS Windows`), but could not be *run* in this
sandbox (no Windows/Wine available) — it shares the exact same source
as the Linux binary that was run, so this is a low-risk gap, not an
unverified one.

### Troubleshooting for a software person

- **"Company not found" / empty ledger list**: company name mismatch,
  or the company isn't currently open in Tally.
- **Connection refused on port 9000**: Step 1's setting needs Tally
  restarted, or another process already holds that port — check with
  `netstat -ano | findstr 9000` on Windows.
- **Works via `curl` on the Tally machine but not from the bridge**: if
  the bridge runs on a different machine than Tally, confirm inbound
  connections on port 9000 aren't firewalled, and set `tallyBaseUrl`
  (setup form) or `TALLY_BASE_URL` (env var) to Tally's LAN IP, not
  `localhost`.
- **Multiple companies open at once**: Tally's gateway serves whichever
  company `SVCURRENTCOMPANY` names — call each analysis once per
  company you need.

See [`README.md`](README.md) for the full architecture and the
honesty note on the XML parsing (built and tested against Tally's
documented format and realistic sample data, not a live instance in
the environment this was built in).
