import type { ChatReply, LoginResult, RunningTrainingJobs, TrainAlgoAiResult } from './types';
export * from './types';
export interface ShikiClientOptions {
    /** e.g. "https://your-host/chatbot" (behind nginx) or "http://localhost:8001" direct. */
    baseUrl: string;
    /** Skip login() if you already have a valid JWT. */
    accessToken?: string;
    /** Override for runtimes without a global fetch (rare — Node 18+, browsers, and React Native all have one). */
    fetchImpl?: typeof fetch;
}
export declare class ShikiApiError extends Error {
    readonly status: number;
    readonly body: unknown;
    constructor(status: number, body: unknown, message: string);
}
export declare class ShikiClient {
    private baseUrl;
    private accessToken;
    private fetchImpl;
    constructor(options: ShikiClientOptions);
    getAccessToken(): string | null;
    setAccessToken(token: string | null): void;
    /** POST /api/auth/login — stores the returned JWT for subsequent calls. */
    login(email: string, password: string): Promise<LoginResult>;
    /**
     * POST /api/chat/message — the one endpoint every ALGO SHIKI client
     * (web, mobile, VS Code) calls for PM Tool, Trading Agent, and
     * chat-triggered Train Algo AI requests alike.
     */
    chat(message: string, sessionId: string, provider?: string | null): Promise<ChatReply>;
    /** Answer a `pendingConfirmation` turn — there is no separate confirm endpoint, just "yes"/"no" on the same session. */
    confirm(sessionId: string, yes: boolean): Promise<ChatReply>;
    /** POST /api/peft-training/link — log a URL as a real training example for `domain`, indexed into that domain's RAG store immediately. */
    trainAlgoAiLink(domain: string, topic: string, url: string): Promise<TrainAlgoAiResult>;
    /** Convenience wrapper over trainAlgoAiFile() for plain text/JSON payloads (e.g. a Tally export) that don't already exist as a file. */
    trainAlgoAiText(domain: string, topic: string, text: string): Promise<TrainAlgoAiResult>;
    /** POST /api/peft-training/upload (multipart) — 5MB max, .txt/.md/.csv/.pdf. */
    trainAlgoAiFile(domain: string, topic: string, filename: string, file: Blob): Promise<TrainAlgoAiResult>;
    /** GET /api/admin/peft-jobs — Admin-only; returns null (not an error) for a non-admin token, matching how the web app hides the badge silently on 403. */
    runningTrainingJobs(): Promise<RunningTrainingJobs | null>;
    private request;
}
