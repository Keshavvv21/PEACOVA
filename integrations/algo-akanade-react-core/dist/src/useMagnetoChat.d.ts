export interface MagnetoChatMessage {
    id: string;
    role: 'user' | 'assistant';
    text: string;
    provider?: string | null;
}
export interface UseMagnetoChatOptions {
    baseUrl: string;
    /** Provide either an existing accessToken, or email+password (the hook logs in lazily on the first send/confirm). */
    email?: string;
    password?: string;
    accessToken?: string;
    sessionId?: string;
    provider?: string | null;
    /** Override for tests, or a runtime without a global fetch. */
    fetchImpl?: typeof fetch;
}
export interface UseMagnetoChatResult {
    messages: MagnetoChatMessage[];
    sending: boolean;
    error: string | null;
    pendingConfirmation: boolean;
    confirmPrompt: string | null;
    send: (text: string) => Promise<void>;
    confirm: (yes: boolean) => Promise<void>;
    reset: () => void;
}
export declare function useMagnetoChat(options: UseMagnetoChatOptions): UseMagnetoChatResult;
