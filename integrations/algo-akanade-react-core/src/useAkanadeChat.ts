// The one piece of chat logic every UI (web widget, RN widget, or a
// consumer's own custom UI) needs: message history, sending a turn,
// answering a pendingConfirmation turn, loading/error state. Uses only
// the plain React hooks API (useState/useCallback/useRef) — no DOM,
// no React Native primitives — so this exact file runs unmodified
// under React DOM and React Native, since both share the same 'react'
// package underneath. algo-akanade-react and algo-akanade-react-native
// each render a different UI on top of this one hook.
import { useCallback, useRef, useState } from 'react';
import { AkanadeClient, type ChatReply } from '@algo-akanade/connector';

export interface AkanadeChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  provider?: string | null;
}

export interface UseAkanadeChatOptions {
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

export interface UseAkanadeChatResult {
  messages: AkanadeChatMessage[];
  sending: boolean;
  error: string | null;
  pendingConfirmation: boolean;
  confirmPrompt: string | null;
  send: (text: string) => Promise<void>;
  confirm: (yes: boolean) => Promise<void>;
  reset: () => void;
}

let messageCounter = 0;
function nextId(): string {
  messageCounter += 1;
  return `m${messageCounter}`;
}

function randomSessionId(): string {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export function useAkanadeChat(options: UseAkanadeChatOptions): UseAkanadeChatResult {
  const { baseUrl, email, password, accessToken, provider = null, fetchImpl } = options;
  const sessionIdRef = useRef(options.sessionId ?? randomSessionId());
  const clientRef = useRef<AkanadeClient | null>(null);
  if (!clientRef.current) {
    clientRef.current = new AkanadeClient({ baseUrl, accessToken, fetchImpl });
  }

  const [messages, setMessages] = useState<AkanadeChatMessage[]>([]);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingConfirmation, setPendingConfirmation] = useState(false);
  const [confirmPrompt, setConfirmPrompt] = useState<string | null>(null);

  const ensureLoggedIn = useCallback(async () => {
    const client = clientRef.current!;
    if (client.getAccessToken()) return;
    if (!email || !password) {
      throw new Error('Not logged in — pass accessToken, or email+password, to useAkanadeChat().');
    }
    await client.login(email, password);
  }, [email, password]);

  const applyReply = useCallback((reply: ChatReply) => {
    setMessages((prev) => [...prev, { id: nextId(), role: 'assistant', text: reply.reply, provider: reply.provider }]);
    setPendingConfirmation(reply.pendingConfirmation);
    setConfirmPrompt(reply.confirmPrompt);
  }, []);

  const send = useCallback(
    async (text: string) => {
      if (!text.trim()) return;
      setSending(true);
      setError(null);
      setMessages((prev) => [...prev, { id: nextId(), role: 'user', text }]);
      try {
        await ensureLoggedIn();
        const reply = await clientRef.current!.chat(text, sessionIdRef.current, provider);
        applyReply(reply);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setSending(false);
      }
    },
    [ensureLoggedIn, applyReply, provider],
  );

  const confirm = useCallback(
    async (yes: boolean) => {
      setSending(true);
      setError(null);
      setMessages((prev) => [...prev, { id: nextId(), role: 'user', text: yes ? 'yes' : 'no' }]);
      try {
        await ensureLoggedIn();
        const reply = await clientRef.current!.confirm(sessionIdRef.current, yes);
        applyReply(reply);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setSending(false);
      }
    },
    [ensureLoggedIn, applyReply],
  );

  const reset = useCallback(() => {
    setMessages([]);
    setPendingConfirmation(false);
    setConfirmPrompt(null);
    setError(null);
  }, []);

  return { messages, sending, error, pendingConfirmation, confirmPrompt, send, confirm, reset };
}
