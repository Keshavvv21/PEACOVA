import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { createBridgeServer } from '../src/server';
import { loadConfig } from '../src/config';

function tempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'algo-tally-server-test-'));
}

function listen(server: http.Server): Promise<number> {
  return new Promise((resolve) => {
    server.listen(0, () => {
      const address = server.address();
      resolve(typeof address === 'object' && address ? address.port : 0);
    });
  });
}

async function fetchJson(port: number, path: string, init?: RequestInit): Promise<{ status: number; body: any }> {
  const res = await fetch(`http://localhost:${port}${path}`, init);
  const text = await res.text();
  return { status: res.status, body: text ? JSON.parse(text) : null };
}

async function fetchText(port: number, path: string, init?: RequestInit): Promise<{ status: number; body: string; location?: string | null }> {
  const res = await fetch(`http://localhost:${port}${path}`, { ...init, redirect: 'manual' });
  return { status: res.status, body: await res.text(), location: res.headers.get('location') };
}

/** A tiny fake ALGO AKANADE backend (real HTTP, no mocking library) so createBridgeServer's real AkanadeClient has something to talk to. */
function startFakeAkanade(): Promise<{ server: http.Server; port: number; sentMessages: string[] }> {
  const sentMessages: string[] = [];
  const server = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
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

test('GET /setup renders the setup form even with nothing configured yet', async () => {
  const bridgeServer = createBridgeServer({ baseDir: tempDir() });
  const port = await listen(bridgeServer);
  try {
    const res = await fetchText(port, '/setup');
    assert.equal(res.status, 200);
    assert.match(res.body, /Algo AI for Tally/);
    assert.match(res.body, /name="algoAkanadeBaseUrl"/);
  } finally {
    bridgeServer.close();
  }
});

test('GET / redirects to /setup when nothing is configured yet', async () => {
  const bridgeServer = createBridgeServer({ baseDir: tempDir() });
  const port = await listen(bridgeServer);
  try {
    const res = await fetchText(port, '/');
    assert.equal(res.status, 302);
    assert.equal(res.location, '/setup');
  } finally {
    bridgeServer.close();
  }
});

test('POST /chat before setup returns a clear error instead of a crash', async () => {
  const bridgeServer = createBridgeServer({ baseDir: tempDir() });
  const port = await listen(bridgeServer);
  try {
    const res = await fetchJson(port, '/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: 'hi' }),
    });
    assert.equal(res.status, 400);
    assert.match(res.body.error, /setup/i);
  } finally {
    bridgeServer.close();
  }
});

test('POST /setup saves the config to disk, and GET / then serves the chat page', async () => {
  const dir = tempDir();
  const bridgeServer = createBridgeServer({ baseDir: dir });
  const port = await listen(bridgeServer);
  try {
    const form = new URLSearchParams({
      algoAkanadeBaseUrl: 'https://example.test/chatbot',
      algoAkanadeEmail: 'a@b.com',
      algoAkanadePassword: 'secret',
      tallyBaseUrl: 'http://localhost:9000',
    });
    const setupRes = await fetchText(port, '/setup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: form.toString(),
    });
    assert.equal(setupRes.status, 200);
    assert.match(setupRes.body, /Saved/);

    const saved = loadConfig(dir);
    assert.equal(saved?.algoAkanadeBaseUrl, 'https://example.test/chatbot');
    assert.equal(saved?.algoAkanadeEmail, 'a@b.com');

    const chatPageRes = await fetchText(port, '/');
    assert.equal(chatPageRes.status, 200);
    assert.match(chatPageRes.body, /Algo AI/);
    assert.match(chatPageRes.body, /id="input"/);
  } finally {
    bridgeServer.close();
  }
});

test('POST /setup with a missing field re-shows the form with an error, and does not save', async () => {
  const dir = tempDir();
  const bridgeServer = createBridgeServer({ baseDir: dir });
  const port = await listen(bridgeServer);
  try {
    const form = new URLSearchParams({ algoAkanadeBaseUrl: 'https://example.test/chatbot', algoAkanadeEmail: '', algoAkanadePassword: 'secret' });
    const res = await fetchText(port, '/setup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: form.toString(),
    });
    assert.equal(res.status, 400);
    assert.match(res.body, /Please fill in every field/);
    assert.equal(loadConfig(dir), null);
  } finally {
    bridgeServer.close();
  }
});

test('after setup, POST /chat logs in once and forwards the message to the real chat endpoint', async () => {
  const { server: fakeAkanade, port: algoAkanadePort, sentMessages } = await startFakeAkanade();
  const dir = tempDir();
  const bridgeServer = createBridgeServer({ baseDir: dir });
  const port = await listen(bridgeServer);
  try {
    const form = new URLSearchParams({
      algoAkanadeBaseUrl: `http://localhost:${algoAkanadePort}`,
      algoAkanadeEmail: 'a@b.com',
      algoAkanadePassword: 'secret',
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

    assert.equal(chatRes.status, 200);
    assert.equal(chatRes.body.reply, 'a real reply');
    assert.deepEqual(sentMessages, ['Backtest AAPL']);
  } finally {
    bridgeServer.close();
    fakeAkanade.close();
  }
});

test('POST /analyze-ledgers surfaces a clear error when Tally is not reachable', async () => {
  const { server: fakeAkanade, port: algoAkanadePort } = await startFakeAkanade();
  const dir = tempDir();
  const bridgeServer = createBridgeServer({ baseDir: dir });
  const port = await listen(bridgeServer);
  try {
    const form = new URLSearchParams({
      algoAkanadeBaseUrl: `http://localhost:${algoAkanadePort}`,
      algoAkanadeEmail: 'a@b.com',
      algoAkanadePassword: 'secret',
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

    assert.equal(res.status, 500);
    assert.match(res.body.error, /Could not reach Tally/);
  } finally {
    bridgeServer.close();
    fakeAkanade.close();
  }
});

test('GET /health reports whether the program has been configured yet', async () => {
  const dir = tempDir();
  const bridgeServer = createBridgeServer({ baseDir: dir });
  const port = await listen(bridgeServer);
  try {
    const before = await fetchJson(port, '/health');
    assert.equal(before.body.configured, false);

    const form = new URLSearchParams({ algoAkanadeBaseUrl: 'https://example.test', algoAkanadeEmail: 'a@b.com', algoAkanadePassword: 'secret' });
    await fetchText(port, '/setup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: form.toString(),
    });

    const after = await fetchJson(port, '/health');
    assert.equal(after.body.configured, true);
  } finally {
    bridgeServer.close();
  }
});
