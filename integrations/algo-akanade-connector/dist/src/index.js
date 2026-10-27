"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ShikiClient = exports.ShikiApiError = void 0;
__exportStar(require("./types"), exports);
class ShikiApiError extends Error {
    constructor(status, body, message) {
        super(message);
        this.status = status;
        this.body = body;
        this.name = 'ShikiApiError';
    }
}
exports.ShikiApiError = ShikiApiError;
class ShikiClient {
    constructor(options) {
        this.baseUrl = options.baseUrl.replace(/\/+$/, '');
        this.accessToken = options.accessToken ?? null;
        const impl = options.fetchImpl ?? globalThis.fetch;
        if (!impl) {
            throw new Error('No fetch implementation available on this runtime — pass fetchImpl explicitly.');
        }
        this.fetchImpl = impl;
    }
    getAccessToken() {
        return this.accessToken;
    }
    setAccessToken(token) {
        this.accessToken = token;
    }
    /** POST /api/auth/login — stores the returned JWT for subsequent calls. */
    async login(email, password) {
        const data = await this.request('POST', '/api/auth/login', { email, password }, { auth: false });
        this.accessToken = data.access_token;
        return { accessToken: this.accessToken };
    }
    /**
     * POST /api/chat/message — the one endpoint every ALGO SHIKI client
     * (web, mobile, VS Code) calls for PM Tool, Trading Agent, and
     * chat-triggered Train Algo AI requests alike.
     */
    async chat(message, sessionId, provider = null) {
        const data = await this.request('POST', '/api/chat/message', { message, session_id: sessionId, provider });
        return {
            reply: data.reply,
            pendingConfirmation: Boolean(data.pending_confirmation),
            confirmPrompt: data.confirm_prompt ?? null,
            provider: data.provider ?? null,
        };
    }
    /** Answer a `pendingConfirmation` turn — there is no separate confirm endpoint, just "yes"/"no" on the same session. */
    async confirm(sessionId, yes) {
        return this.chat(yes ? 'yes' : 'no', sessionId);
    }
    /** POST /api/peft-training/link — log a URL as a real training example for `domain`, indexed into that domain's RAG store immediately. */
    async trainAlgoAiLink(domain, topic, url) {
        const data = await this.request('POST', '/api/peft-training/link', { domain, topic, url });
        return toTrainResult(data);
    }
    /** Convenience wrapper over trainAlgoAiFile() for plain text/JSON payloads (e.g. a Tally export) that don't already exist as a file. */
    async trainAlgoAiText(domain, topic, text) {
        return this.trainAlgoAiFile(domain, topic, `${topic || 'note'}.txt`, new Blob([text], { type: 'text/plain' }));
    }
    /** POST /api/peft-training/upload (multipart) — 5MB max, .txt/.md/.csv/.pdf. */
    async trainAlgoAiFile(domain, topic, filename, file) {
        const form = new FormData();
        form.append('domain', domain);
        form.append('topic', topic);
        form.append('file', file, filename);
        const data = await this.request('POST', '/api/peft-training/upload', form);
        return toTrainResult(data);
    }
    /** GET /api/admin/peft-jobs — Admin-only; returns null (not an error) for a non-admin token, matching how the web app hides the badge silently on 403. */
    async runningTrainingJobs() {
        try {
            const data = await this.request('GET', '/api/admin/peft-jobs');
            return {
                jobs: data.jobs ?? [],
                running: data.running ?? [],
            };
        }
        catch (err) {
            if (err instanceof ShikiApiError && err.status === 403)
                return null;
            throw err;
        }
    }
    async request(method, path, body, opts = {}) {
        const auth = opts.auth ?? true;
        const headers = {};
        if (auth) {
            if (!this.accessToken)
                throw new Error('Not logged in — call login() first (or pass accessToken in the constructor).');
            headers.Authorization = `Bearer ${this.accessToken}`;
        }
        let payload;
        if (typeof FormData !== 'undefined' && body instanceof FormData) {
            payload = body;
        }
        else if (body !== undefined) {
            headers['Content-Type'] = 'application/json';
            payload = JSON.stringify(body);
        }
        const res = await this.fetchImpl(`${this.baseUrl}${path}`, { method, headers, body: payload });
        const text = await res.text();
        const data = text ? JSON.parse(text) : {};
        if (!res.ok) {
            const detail = data && typeof data === 'object' && 'detail' in data
                ? String(data.detail)
                : res.statusText;
            throw new ShikiApiError(res.status, data, `ALGO SHIKI API ${method} ${path} failed (${res.status}): ${detail}`);
        }
        return data;
    }
}
exports.ShikiClient = ShikiClient;
function toTrainResult(data) {
    return {
        domain: data.domain,
        topic: data.topic,
        source: data.source,
        charsLogged: data.chars_logged,
        ragChunksIndexed: data.rag_chunks_indexed ?? 0,
        jobStatusNote: data.job_status_note ?? null,
    };
}
