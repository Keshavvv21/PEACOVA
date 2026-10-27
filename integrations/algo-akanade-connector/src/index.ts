// A plain client module for ALGO AKANADE's existing REST API — not a
// separate "SDK product", just the few functions any external caller
// (a Tally bridge, a React app, a React Native app, an ERP scripting
// layer) needs around the endpoints documented in
// orchestrator_framework_layer/akanade/README_EXTERNAL_API.md. No new
// backend endpoints were added for this — it wraps what already ships.
import type { ChatReply, LoginResult, RunningTrainingJobs, TrainAlgoAiResult } from './types';

export * from './types';

export interface AkanadeClientOptions {
  /** e.g. "https://your-host/chatbot" (behind nginx) or "http://localhost:8001" direct. */
  baseUrl: string;
  /** Skip login() if you already have a valid JWT. */
  accessToken?: string;
  /** Override for runtimes without a global fetch (rare — Node 18+, browsers, and React Native all have one). */
  fetchImpl?: typeof fetch;
}

export class AkanadeApiError extends Error {
  constructor(public readonly status: number, public readonly body: unknown, message: string) {
    super(message);
    this.name = 'AkanadeApiError';
  }
}

export class AkanadeClient {
  private baseUrl: string;
  private accessToken: string | null;
  private fetchImpl: typeof fetch;

  constructor(options: AkanadeClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, '');
    this.accessToken = options.accessToken ?? null;
    const impl = options.fetchImpl ?? (globalThis as { fetch?: typeof fetch }).fetch;
    if (!impl) {
      throw new Error('No fetch implementation available on this runtime — pass fetchImpl explicitly.');
    }
    this.fetchImpl = impl;
  }

  getAccessToken(): string | null {
    return this.accessToken;
  }

  setAccessToken(token: string | null): void {
    this.accessToken = token;
  }

  /** POST /api/auth/login — stores the returned JWT for subsequent calls. */
  async login(email: string, password: string): Promise<LoginResult> {
    const data = await this.request('POST', '/api/auth/login', { email, password }, { auth: false });
    this.accessToken = data.access_token as string;
    return { accessToken: this.accessToken };
  }

  /**
   * POST /api/chat/message — the one endpoint every ALGO AKANADE client
   * (web, mobile, VS Code) calls for PM Tool, Trading Agent, and
   * chat-triggered Train Algo AI requests alike.
   */
  async chat(message: string, sessionId: string, provider: string | null = null): Promise<ChatReply> {
    const data = await this.request('POST', '/api/chat/message', { message, session_id: sessionId, provider });
    return {
      reply: data.reply as string,
      pendingConfirmation: Boolean(data.pending_confirmation),
      confirmPrompt: (data.confirm_prompt as string | undefined) ?? null,
      provider: (data.provider as string | undefined) ?? null,
    };
  }

  /** Answer a `pendingConfirmation` turn — there is no separate confirm endpoint, just "yes"/"no" on the same session. */
  async confirm(sessionId: string, yes: boolean): Promise<ChatReply> {
    return this.chat(yes ? 'yes' : 'no', sessionId);
  }

  /** POST /api/peft-training/link — log a URL as a real training example for `domain`, indexed into that domain's RAG store immediately. */
  async trainAlgoAiLink(domain: string, topic: string, url: string): Promise<TrainAlgoAiResult> {
    const data = await this.request('POST', '/api/peft-training/link', { domain, topic, url });
    return toTrainResult(data);
  }

  /** Convenience wrapper over trainAlgoAiFile() for plain text/JSON payloads (e.g. a Tally export) that don't already exist as a file. */
  async trainAlgoAiText(domain: string, topic: string, text: string): Promise<TrainAlgoAiResult> {
    return this.trainAlgoAiFile(domain, topic, `${topic || 'note'}.txt`, new Blob([text], { type: 'text/plain' }));
  }

  /** POST /api/peft-training/upload (multipart) — 5MB max, .txt/.md/.csv/.pdf. */
  async trainAlgoAiFile(domain: string, topic: string, filename: string, file: Blob): Promise<TrainAlgoAiResult> {
    const form = new FormData();
    form.append('domain', domain);
    form.append('topic', topic);
    form.append('file', file, filename);
    const data = await this.request('POST', '/api/peft-training/upload', form);
    return toTrainResult(data);
  }

  /** GET /api/admin/peft-jobs — Admin-only; returns null (not an error) for a non-admin token, matching how the web app hides the badge silently on 403. */
  async runningTrainingJobs(): Promise<RunningTrainingJobs | null> {
    try {
      const data = await this.request('GET', '/api/admin/peft-jobs');
      return {
        jobs: (data.jobs as Array<Record<string, unknown>> | undefined) ?? [],
        running: (data.running as string[] | undefined) ?? [],
      };
    } catch (err) {
      if (err instanceof AkanadeApiError && err.status === 403) return null;
      throw err;
    }
  }

  private async request(
    method: string,
    path: string,
    body?: unknown,
    opts: { auth?: boolean } = {},
  ): Promise<Record<string, unknown>> {
    const auth = opts.auth ?? true;
    const headers: Record<string, string> = {};
    if (auth) {
      if (!this.accessToken) throw new Error('Not logged in — call login() first (or pass accessToken in the constructor).');
      headers.Authorization = `Bearer ${this.accessToken}`;
    }

    let payload: BodyInit | undefined;
    if (typeof FormData !== 'undefined' && body instanceof FormData) {
      payload = body;
    } else if (body !== undefined) {
      headers['Content-Type'] = 'application/json';
      payload = JSON.stringify(body);
    }

    const res = await this.fetchImpl(`${this.baseUrl}${path}`, { method, headers, body: payload });
    const text = await res.text();
    const data: unknown = text ? JSON.parse(text) : {};

    if (!res.ok) {
      const detail =
        data && typeof data === 'object' && 'detail' in (data as Record<string, unknown>)
          ? String((data as Record<string, unknown>).detail)
          : res.statusText;
      throw new AkanadeApiError(res.status, data, `ALGO AKANADE API ${method} ${path} failed (${res.status}): ${detail}`);
    }
    return data as Record<string, unknown>;
  }
}

function toTrainResult(data: Record<string, unknown>): TrainAlgoAiResult {
  return {
    domain: data.domain as string,
    topic: data.topic as string,
    source: data.source as string,
    charsLogged: data.chars_logged as number,
    ragChunksIndexed: (data.rag_chunks_indexed as number | undefined) ?? 0,
    jobStatusNote: (data.job_status_note as string | undefined) ?? null,
  };
}
