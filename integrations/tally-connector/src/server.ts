// A small local program meant to run on the same machine as Tally (or
// one that can reach Tally's gateway over the LAN). Two front doors:
//
// 1. A browser popup chat (GET /, POST /chat, plus the Tally-aware
//    /analyze-ledgers and /analyze-daybook quick actions) — for a
//    non-technical user, this is the entire product: open the program,
//    a chat window appears.
// 2. Tally's own TDL `HTTP Post` action (built in since Tally.ERP 9
//    Series 6.x) can call the same /analyze-ledgers etc. routes
//    directly from a custom Tally menu item — see ../tdl/AlgoAkanadeAI.tdl.
//
// Nothing here requires Tally or the user to know anything about
// ALGO AKANADE's own API/auth — that's handled once, in the setup page.
import http from 'node:http';
import { AkanadeClient } from '@algo-akanade/connector';
import { renderChatPage } from './chatPage';
import { type BridgeConfig, DEFAULT_BRIDGE_PORT, loadConfig, saveConfig } from './config';
import { renderSetupPage } from './setupPage';
import { TallyAkanadeBridge } from './bridge';
import { TallyGatewayClient } from './tallyGateway';

export interface CreateBridgeServerOptions {
  /** Directory the config file lives in/is written to — usually the folder the program itself runs from. */
  baseDir: string;
}

async function readBody(req: http.IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks).toString('utf8');
}

function sendJson(res: http.ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(text) });
  res.end(text);
}

function sendHtml(res: http.ServerResponse, status: number, html: string): void {
  res.writeHead(status, { 'Content-Type': 'text/html; charset=utf-8', 'Content-Length': Buffer.byteLength(html) });
  res.end(html);
}

function parseJsonBody(raw: string): Record<string, unknown> {
  if (!raw.trim()) return {};
  return JSON.parse(raw) as Record<string, unknown>;
}

function parseFormBody(raw: string): Record<string, string> {
  const params = new URLSearchParams(raw);
  return Object.fromEntries(params.entries());
}

export function createBridgeServer(options: CreateBridgeServerOptions): http.Server {
  let config: BridgeConfig | null = loadConfig(options.baseDir);
  let akanade: AkanadeClient | null = null;
  let bridge: TallyAkanadeBridge | null = null;

  function applyConfig(next: BridgeConfig): void {
    config = next;
    akanade = new AkanadeClient({ baseUrl: next.algoAkanadeBaseUrl });
    bridge = new TallyAkanadeBridge({ akanade, tally: new TallyGatewayClient({ baseUrl: next.tallyBaseUrl }) });
  }

  if (config) applyConfig(config);

  async function ensureLoggedIn(): Promise<void> {
    if (!akanade || !config) throw new Error('Not configured yet — open /setup first.');
    if (!akanade.getAccessToken()) {
      await akanade.login(config.algoAkanadeEmail, config.algoAkanadePassword);
    }
  }

  async function handle(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const url = new URL(req.url ?? '/', 'http://localhost');

    if (req.method === 'GET' && url.pathname === '/health') {
      sendJson(res, 200, { ok: true, configured: Boolean(config) });
      return;
    }

    if (req.method === 'GET' && url.pathname === '/setup') {
      sendHtml(res, 200, renderSetupPage(config ?? {}));
      return;
    }

    if (req.method === 'POST' && url.pathname === '/setup') {
      const body = parseFormBody(await readBody(req));
      const next: BridgeConfig = {
        algoAkanadeBaseUrl: (body.algoAkanadeBaseUrl ?? '').trim().replace(/\/+$/, ''),
        algoAkanadeEmail: (body.algoAkanadeEmail ?? '').trim(),
        algoAkanadePassword: body.algoAkanadePassword || config?.algoAkanadePassword || '',
        tallyBaseUrl: (body.tallyBaseUrl ?? '').trim() || 'http://localhost:9000',
        bridgePort: config?.bridgePort ?? DEFAULT_BRIDGE_PORT,
      };
      if (!next.algoAkanadeBaseUrl || !next.algoAkanadeEmail || !next.algoAkanadePassword) {
        sendHtml(res, 400, renderSetupPage(next, { error: 'Please fill in every field.' }));
        return;
      }
      saveConfig(options.baseDir, next);
      applyConfig(next);
      sendHtml(res, 200, renderSetupPage(next, { saved: true }));
      return;
    }

    if (req.method === 'GET' && (url.pathname === '/' || url.pathname === '/chat')) {
      if (!config) {
        res.writeHead(302, { Location: '/setup' });
        res.end();
        return;
      }
      sendHtml(res, 200, renderChatPage());
      return;
    }

    if (req.method === 'POST' && url.pathname === '/chat') {
      if (!config || !akanade) {
        sendJson(res, 400, { error: 'Not configured yet — open /setup first.' });
        return;
      }
      const body = parseJsonBody(await readBody(req));
      try {
        await ensureLoggedIn();
        const reply = await akanade.chat(
          String(body.message ?? ''),
          String(body.sessionId ?? 'tally-popup'),
          (body.provider as string | null | undefined) ?? null,
        );
        sendJson(res, 200, { reply: reply.reply, pendingConfirmation: reply.pendingConfirmation, confirmPrompt: reply.confirmPrompt });
      } catch (err) {
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
        const reply = await bridge.analyzeLedgers(
          String(body.companyName ?? ''),
          String(body.question ?? ''),
          body.sessionId ? String(body.sessionId) : undefined,
        );
        sendJson(res, 200, { reply });
      } catch (err) {
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
        const reply = await bridge.analyzeDayBook(
          String(body.companyName ?? ''),
          String(body.fromDate ?? ''),
          String(body.toDate ?? ''),
          String(body.question ?? ''),
          body.sessionId ? String(body.sessionId) : undefined,
        );
        sendJson(res, 200, { reply });
      } catch (err) {
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
        const result = await bridge.trainOnLedgers(
          String(body.companyName ?? ''),
          String(body.domain ?? 'general'),
          String(body.topic ?? 'Tally ledger export'),
        );
        sendJson(res, 200, result);
      } catch (err) {
        sendJson(res, 500, { error: err instanceof Error ? err.message : String(err) });
      }
      return;
    }

    sendJson(res, 404, { error: `No such route: ${req.method} ${url.pathname}` });
  }

  return http.createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      sendJson(res, 500, { error: err instanceof Error ? err.message : String(err) });
    });
  });
}
