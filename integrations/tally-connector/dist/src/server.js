"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.createBridgeServer = createBridgeServer;
// A small local program meant to run on the same machine as Tally (or
// one that can reach Tally's gateway over the LAN). Two front doors:
//
// 1. A browser popup chat (GET /, POST /chat, plus the Tally-aware
//    /analyze-ledgers and /analyze-daybook quick actions) — for a
//    non-technical user, this is the entire product: open the program,
//    a chat window appears.
// 2. Tally's own TDL `HTTP Post` action (built in since Tally.ERP 9
//    Series 6.x) can call the same /analyze-ledgers etc. routes
//    directly from a custom Tally menu item — see ../tdl/AlgoMagnetoAI.tdl.
//
// Nothing here requires Tally or the user to know anything about
// MAGNETO's own API/auth — that's handled once, in the setup page.
const node_http_1 = __importDefault(require("node:http"));
const connector_1 = require("@algo-magneto/connector");
const chatPage_1 = require("./chatPage");
const config_1 = require("./config");
const setupPage_1 = require("./setupPage");
const bridge_1 = require("./bridge");
const tallyGateway_1 = require("./tallyGateway");
async function readBody(req) {
    const chunks = [];
    for await (const chunk of req)
        chunks.push(chunk);
    return Buffer.concat(chunks).toString('utf8');
}
function sendJson(res, status, body) {
    const text = JSON.stringify(body);
    res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(text) });
    res.end(text);
}
function sendHtml(res, status, html) {
    res.writeHead(status, { 'Content-Type': 'text/html; charset=utf-8', 'Content-Length': Buffer.byteLength(html) });
    res.end(html);
}
function parseJsonBody(raw) {
    if (!raw.trim())
        return {};
    return JSON.parse(raw);
}
function parseFormBody(raw) {
    const params = new URLSearchParams(raw);
    return Object.fromEntries(params.entries());
}
function createBridgeServer(options) {
    let config = (0, config_1.loadConfig)(options.baseDir);
    let magneto = null;
    let bridge = null;
    function applyConfig(next) {
        config = next;
        magneto = new connector_1.MagnetoClient({ baseUrl: next.magnetoBaseUrl });
        bridge = new bridge_1.TallyMagnetoBridge({ magneto, tally: new tallyGateway_1.TallyGatewayClient({ baseUrl: next.tallyBaseUrl }) });
    }
    if (config)
        applyConfig(config);
    async function ensureLoggedIn() {
        if (!magneto || !config)
            throw new Error('Not configured yet — open /setup first.');
        if (!magneto.getAccessToken()) {
            await magneto.login(config.magnetoEmail, config.magnetoPassword);
        }
    }
    async function handle(req, res) {
        const url = new URL(req.url ?? '/', 'http://localhost');
        if (req.method === 'GET' && url.pathname === '/health') {
            sendJson(res, 200, { ok: true, configured: Boolean(config) });
            return;
        }
        if (req.method === 'GET' && url.pathname === '/setup') {
            sendHtml(res, 200, (0, setupPage_1.renderSetupPage)(config ?? {}));
            return;
        }
        if (req.method === 'POST' && url.pathname === '/setup') {
            const body = parseFormBody(await readBody(req));
            const next = {
                magnetoBaseUrl: (body.magnetoBaseUrl ?? '').trim().replace(/\/+$/, ''),
                magnetoEmail: (body.magnetoEmail ?? '').trim(),
                magnetoPassword: body.magnetoPassword || config?.magnetoPassword || '',
                tallyBaseUrl: (body.tallyBaseUrl ?? '').trim() || 'http://localhost:9000',
                bridgePort: config?.bridgePort ?? config_1.DEFAULT_BRIDGE_PORT,
            };
            if (!next.magnetoBaseUrl || !next.magnetoEmail || !next.magnetoPassword) {
                sendHtml(res, 400, (0, setupPage_1.renderSetupPage)(next, { error: 'Please fill in every field.' }));
                return;
            }
            (0, config_1.saveConfig)(options.baseDir, next);
            applyConfig(next);
            sendHtml(res, 200, (0, setupPage_1.renderSetupPage)(next, { saved: true }));
            return;
        }
        if (req.method === 'GET' && (url.pathname === '/' || url.pathname === '/chat')) {
            if (!config) {
                res.writeHead(302, { Location: '/setup' });
                res.end();
                return;
            }
            sendHtml(res, 200, (0, chatPage_1.renderChatPage)());
            return;
        }
        if (req.method === 'POST' && url.pathname === '/chat') {
            if (!config || !magneto) {
                sendJson(res, 400, { error: 'Not configured yet — open /setup first.' });
                return;
            }
            const body = parseJsonBody(await readBody(req));
            try {
                await ensureLoggedIn();
                const reply = await magneto.chat(String(body.message ?? ''), String(body.sessionId ?? 'tally-popup'), body.provider ?? null);
                sendJson(res, 200, { reply: reply.reply, pendingConfirmation: reply.pendingConfirmation, confirmPrompt: reply.confirmPrompt });
            }
            catch (err) {
                sendJson(res, 500, { error: err instanceof Error ? err.message : String(err) });
            }
            return;
        }
        if (req.method === 'POST' && url.pathname === '/analyze-ledgers') {
            if (!config || !bridge) {
                sendJson(res, 400, { error: 'Not configured yet — open /setup first.' });
                return;
            }
            const body = parseJsonBody(await readBody(req));
            try {
                await ensureLoggedIn();
                const reply = await bridge.analyzeLedgers(String(body.companyName ?? ''), String(body.question ?? ''), body.sessionId ? String(body.sessionId) : undefined);
                sendJson(res, 200, { reply });
            }
            catch (err) {
                sendJson(res, 500, { error: err instanceof Error ? err.message : String(err) });
            }
            return;
        }
        if (req.method === 'POST' && url.pathname === '/analyze-daybook') {
            if (!config || !bridge) {
                sendJson(res, 400, { error: 'Not configured yet — open /setup first.' });
                return;
            }
            const body = parseJsonBody(await readBody(req));
            try {
                await ensureLoggedIn();
                const reply = await bridge.analyzeDayBook(String(body.companyName ?? ''), String(body.fromDate ?? ''), String(body.toDate ?? ''), String(body.question ?? ''), body.sessionId ? String(body.sessionId) : undefined);
                sendJson(res, 200, { reply });
            }
            catch (err) {
                sendJson(res, 500, { error: err instanceof Error ? err.message : String(err) });
            }
            return;
        }
        if (req.method === 'POST' && url.pathname === '/train-on-ledgers') {
            if (!config || !bridge) {
                sendJson(res, 400, { error: 'Not configured yet — open /setup first.' });
                return;
            }
            const body = parseJsonBody(await readBody(req));
            try {
                await ensureLoggedIn();
                const result = await bridge.trainOnLedgers(String(body.companyName ?? ''), String(body.domain ?? 'general'), String(body.topic ?? 'Tally ledger export'));
                sendJson(res, 200, result);
            }
            catch (err) {
                sendJson(res, 500, { error: err instanceof Error ? err.message : String(err) });
            }
            return;
        }
        sendJson(res, 404, { error: `No such route: ${req.method} ${url.pathname}` });
    }
    return node_http_1.default.createServer((req, res) => {
        void handle(req, res).catch((err) => {
            sendJson(res, 500, { error: err instanceof Error ? err.message : String(err) });
        });
    });
}
