"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const strict_1 = __importDefault(require("node:assert/strict"));
const node_fs_1 = __importDefault(require("node:fs"));
const node_http_1 = __importDefault(require("node:http"));
const node_os_1 = __importDefault(require("node:os"));
const node_path_1 = __importDefault(require("node:path"));
const node_test_1 = require("node:test");
const server_1 = require("../src/server");
const config_1 = require("../src/config");
function tempDir() {
    return node_fs_1.default.mkdtempSync(node_path_1.default.join(node_os_1.default.tmpdir(), 'algo-tally-server-test-'));
}
function listen(server) {
    return new Promise((resolve) => {
        server.listen(0, () => {
            const address = server.address();
            resolve(typeof address === 'object' && address ? address.port : 0);
        });
    });
}
async function fetchJson(port, path, init) {
    const res = await fetch(`http://localhost:${port}${path}`, init);
    const text = await res.text();
    return { status: res.status, body: text ? JSON.parse(text) : null };
}
async function fetchText(port, path, init) {
    const res = await fetch(`http://localhost:${port}${path}`, { ...init, redirect: 'manual' });
    return { status: res.status, body: await res.text(), location: res.headers.get('location') };
}
/** A tiny fake MAGNETO backend (real HTTP, no mocking library) so createBridgeServer's real MagnetoClient has something to talk to. */
function startFakeMagneto() {
    const sentMessages = [];
    const server = node_http_1.default.createServer((req, res) => {
        const chunks = [];
        req.on('data', (c) => chunks.push(c));
        req.on('end', () => {
            const body = Buffer.concat(chunks).toString('utf8');
            res.setHeader('Content-Type', 'application/json');
            if (req.url === '/api/auth/login') {
                res.end(JSON.stringify({ access_token: 'tok-123' }));
                return;
            }
            if (req.url === '/api/chat/message') {
                const parsed = JSON.parse(body || '{}');
                sentMessages.push(parsed.message);
                res.end(JSON.stringify({ reply: 'a real reply', pending_confirmation: false, confirm_prompt: null, provider: 'groq' }));
                return;
            }
            res.statusCode = 404;
            res.end(JSON.stringify({ detail: 'not found' }));
        });
    });
    return new Promise((resolve) => {
        server.listen(0, () => {
            const address = server.address();
            const port = typeof address === 'object' && address ? address.port : 0;
            resolve({ server, port, sentMessages });
        });
    });
}
(0, node_test_1.test)('GET /setup renders the setup form even with nothing configured yet', async () => {
    const bridgeServer = (0, server_1.createBridgeServer)({ baseDir: tempDir() });
    const port = await listen(bridgeServer);
    try {
        const res = await fetchText(port, '/setup');
        strict_1.default.equal(res.status, 200);
        strict_1.default.match(res.body, /Algo AI for Tally/);
        strict_1.default.match(res.body, /name="magnetoBaseUrl"/);
    }
    finally {
        bridgeServer.close();
    }
});
(0, node_test_1.test)('GET / redirects to /setup when nothing is configured yet', async () => {
    const bridgeServer = (0, server_1.createBridgeServer)({ baseDir: tempDir() });
    const port = await listen(bridgeServer);
    try {
        const res = await fetchText(port, '/');
        strict_1.default.equal(res.status, 302);
        strict_1.default.equal(res.location, '/setup');
    }
    finally {
        bridgeServer.close();
    }
});
(0, node_test_1.test)('POST /chat before setup returns a clear error instead of a crash', async () => {
    const bridgeServer = (0, server_1.createBridgeServer)({ baseDir: tempDir() });
    const port = await listen(bridgeServer);
    try {
        const res = await fetchJson(port, '/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: 'hi' }),
        });
        strict_1.default.equal(res.status, 400);
        strict_1.default.match(res.body.error, /setup/i);
    }
    finally {
        bridgeServer.close();
    }
});
(0, node_test_1.test)('POST /setup saves the config to disk, and GET / then serves the chat page', async () => {
    const dir = tempDir();
    const bridgeServer = (0, server_1.createBridgeServer)({ baseDir: dir });
    const port = await listen(bridgeServer);
    try {
        const form = new URLSearchParams({
            magnetoBaseUrl: 'https://example.test/chatbot',
            magnetoEmail: 'a@b.com',
            magnetoPassword: 'secret',
            tallyBaseUrl: 'http://localhost:9000',
        });
        const setupRes = await fetchText(port, '/setup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: form.toString(),
        });
        strict_1.default.equal(setupRes.status, 200);
        strict_1.default.match(setupRes.body, /Saved/);
        const saved = (0, config_1.loadConfig)(dir);
        strict_1.default.equal(saved?.magnetoBaseUrl, 'https://example.test/chatbot');
        strict_1.default.equal(saved?.magnetoEmail, 'a@b.com');
        const chatPageRes = await fetchText(port, '/');
        strict_1.default.equal(chatPageRes.status, 200);
        strict_1.default.match(chatPageRes.body, /Algo AI/);
        strict_1.default.match(chatPageRes.body, /id="input"/);
    }
    finally {
        bridgeServer.close();
    }
});
(0, node_test_1.test)('POST /setup with a missing field re-shows the form with an error, and does not save', async () => {
    const dir = tempDir();
    const bridgeServer = (0, server_1.createBridgeServer)({ baseDir: dir });
    const port = await listen(bridgeServer);
    try {
        const form = new URLSearchParams({ magnetoBaseUrl: 'https://example.test/chatbot', magnetoEmail: '', magnetoPassword: 'secret' });
        const res = await fetchText(port, '/setup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: form.toString(),
        });
        strict_1.default.equal(res.status, 400);
        strict_1.default.match(res.body, /Please fill in every field/);
        strict_1.default.equal((0, config_1.loadConfig)(dir), null);
    }
    finally {
        bridgeServer.close();
    }
});
(0, node_test_1.test)('after setup, POST /chat logs in once and forwards the message to the real chat endpoint', async () => {
    const { server: fakeMagneto, port: magnetoPort, sentMessages } = await startFakeMagneto();
    const dir = tempDir();
    const bridgeServer = (0, server_1.createBridgeServer)({ baseDir: dir });
    const port = await listen(bridgeServer);
    try {
        const form = new URLSearchParams({
            magnetoBaseUrl: `http://localhost:${magnetoPort}`,
            magnetoEmail: 'a@b.com',
            magnetoPassword: 'secret',
            tallyBaseUrl: 'http://localhost:9000',
        });
        await fetchText(port, '/setup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: form.toString(),
        });
        const chatRes = await fetchJson(port, '/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: 'Backtest AAPL', sessionId: 's1' }),
        });
        strict_1.default.equal(chatRes.status, 200);
        strict_1.default.equal(chatRes.body.reply, 'a real reply');
        strict_1.default.deepEqual(sentMessages, ['Backtest AAPL']);
    }
    finally {
        bridgeServer.close();
        fakeMagneto.close();
    }
});
(0, node_test_1.test)('POST /analyze-ledgers surfaces a clear error when Tally is not reachable', async () => {
    const { server: fakeMagneto, port: magnetoPort } = await startFakeMagneto();
    const dir = tempDir();
    const bridgeServer = (0, server_1.createBridgeServer)({ baseDir: dir });
    const port = await listen(bridgeServer);
    try {
        const form = new URLSearchParams({
            magnetoBaseUrl: `http://localhost:${magnetoPort}`,
            magnetoEmail: 'a@b.com',
            magnetoPassword: 'secret',
            // Deliberately an unreachable port — no Tally is running in this test.
            tallyBaseUrl: 'http://localhost:1',
        });
        await fetchText(port, '/setup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: form.toString(),
        });
        const res = await fetchJson(port, '/analyze-ledgers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ companyName: 'My Company', question: 'anything?' }),
        });
        strict_1.default.equal(res.status, 500);
        strict_1.default.match(res.body.error, /Could not reach Tally/);
    }
    finally {
        bridgeServer.close();
        fakeMagneto.close();
    }
});
(0, node_test_1.test)('GET /health reports whether the program has been configured yet', async () => {
    const dir = tempDir();
    const bridgeServer = (0, server_1.createBridgeServer)({ baseDir: dir });
    const port = await listen(bridgeServer);
    try {
        const before = await fetchJson(port, '/health');
        strict_1.default.equal(before.body.configured, false);
        const form = new URLSearchParams({ magnetoBaseUrl: 'https://example.test', magnetoEmail: 'a@b.com', magnetoPassword: 'secret' });
        await fetchText(port, '/setup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: form.toString(),
        });
        const after = await fetchJson(port, '/health');
        strict_1.default.equal(after.body.configured, true);
    }
    finally {
        bridgeServer.close();
    }
});
