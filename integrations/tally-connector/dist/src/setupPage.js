"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.renderSetupPage = renderSetupPage;
const config_1 = require("./config");
function esc(value) {
    return (value ?? '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
}
function renderSetupPage(existing = {}, opts = {}) {
    return `<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Algo AI for Tally — Setup</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 480px; margin: 48px auto; padding: 0 16px; color: #111; background: #fafafa; }
  h1 { font-size: 20px; margin-bottom: 4px; }
  p.sub { color: #555; margin-top: 0; }
  label { display: block; margin-top: 16px; font-weight: 600; font-size: 14px; }
  input { width: 100%; padding: 9px 10px; margin-top: 5px; border: 1px solid #ccc; border-radius: 6px; box-sizing: border-box; font-size: 14px; }
  button { margin-top: 22px; padding: 11px 20px; border: none; border-radius: 6px; background: #1a56db; color: #fff; font-weight: 600; font-size: 14px; cursor: pointer; }
  button:hover { background: #1543ad; }
  .note { color: #666; font-size: 12px; margin-top: 5px; }
  .banner { padding: 10px 12px; border-radius: 6px; margin-bottom: 18px; font-size: 14px; }
  .banner.ok { background: #dcfce7; color: #166534; }
  .banner.err { background: #fee2e2; color: #991b1b; }
  a { color: #1a56db; }
</style>
</head>
<body>
  <h1>Algo AI for Tally</h1>
  <p class="sub">One-time setup. Everything below is saved only on this computer.</p>
  ${opts.saved ? '<div class="banner ok">Saved — Algo AI is running. <a href="/">Open the chat</a>.</div>' : ''}
  ${opts.error ? `<div class="banner err">${esc(opts.error)}</div>` : ''}
  <form method="POST" action="/setup">
    <label for="magnetoBaseUrl">Your Algo AI web address</label>
    <input id="magnetoBaseUrl" name="magnetoBaseUrl" placeholder="https://your-host/chatbot" value="${esc(existing.magnetoBaseUrl)}" required />
    <div class="note">The same address you already use to open Algo AI in a browser.</div>

    <label for="magnetoEmail">Your email</label>
    <input id="magnetoEmail" name="magnetoEmail" type="email" value="${esc(existing.magnetoEmail)}" required />

    <label for="magnetoPassword">Your password</label>
    <input id="magnetoPassword" name="magnetoPassword" type="password" required />
    <div class="note">Stored in a file on this computer only — never sent anywhere except the address above.</div>

    <label for="tallyBaseUrl">Tally's connection address</label>
    <input id="tallyBaseUrl" name="tallyBaseUrl" value="${esc(existing.tallyBaseUrl) || config_1.DEFAULT_TALLY_BASE_URL}" />
    <div class="note">Leave this as it is unless you were told to change Tally's port.</div>

    <button type="submit">Save and start</button>
  </form>
</body>
</html>`;
}
