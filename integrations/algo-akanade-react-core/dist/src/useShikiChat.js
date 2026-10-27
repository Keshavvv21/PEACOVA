"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.useShikiChat = useShikiChat;
// The one piece of chat logic every UI (web widget, RN widget, or a
// consumer's own custom UI) needs: message history, sending a turn,
// answering a pendingConfirmation turn, loading/error state. Uses only
// the plain React hooks API (useState/useCallback/useRef) — no DOM,
// no React Native primitives — so this exact file runs unmodified
// under React DOM and React Native, since both share the same 'react'
// package underneath. algo-shiki-react and algo-shiki-react-native
// each render a different UI on top of this one hook.
const react_1 = require("react");
const connector_1 = require("@algo-shiki/connector");
let messageCounter = 0;
function nextId() {
    messageCounter += 1;
    return `m${messageCounter}`;
}
function randomSessionId() {
    return `session-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}
function useShikiChat(options) {
    const { baseUrl, email, password, accessToken, provider = null, fetchImpl } = options;
    const sessionIdRef = (0, react_1.useRef)(options.sessionId ?? randomSessionId());
    const clientRef = (0, react_1.useRef)(null);
    if (!clientRef.current) {
        clientRef.current = new connector_1.ShikiClient({ baseUrl, accessToken, fetchImpl });
    }
    const [messages, setMessages] = (0, react_1.useState)([]);
    const [sending, setSending] = (0, react_1.useState)(false);
    const [error, setError] = (0, react_1.useState)(null);
    const [pendingConfirmation, setPendingConfirmation] = (0, react_1.useState)(false);
    const [confirmPrompt, setConfirmPrompt] = (0, react_1.useState)(null);
    const ensureLoggedIn = (0, react_1.useCallback)(async () => {
        const client = clientRef.current;
        if (client.getAccessToken())
            return;
        if (!email || !password) {
            throw new Error('Not logged in — pass accessToken, or email+password, to useShikiChat().');
        }
        await client.login(email, password);
    }, [email, password]);
    const applyReply = (0, react_1.useCallback)((reply) => {
        setMessages((prev) => [...prev, { id: nextId(), role: 'assistant', text: reply.reply, provider: reply.provider }]);
        setPendingConfirmation(reply.pendingConfirmation);
        setConfirmPrompt(reply.confirmPrompt);
    }, []);
    const send = (0, react_1.useCallback)(async (text) => {
        if (!text.trim())
            return;
        setSending(true);
        setError(null);
        setMessages((prev) => [...prev, { id: nextId(), role: 'user', text }]);
        try {
            await ensureLoggedIn();
            const reply = await clientRef.current.chat(text, sessionIdRef.current, provider);
            applyReply(reply);
        }
        catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
        finally {
            setSending(false);
        }
    }, [ensureLoggedIn, applyReply, provider]);
    const confirm = (0, react_1.useCallback)(async (yes) => {
        setSending(true);
        setError(null);
        setMessages((prev) => [...prev, { id: nextId(), role: 'user', text: yes ? 'yes' : 'no' }]);
        try {
            await ensureLoggedIn();
            const reply = await clientRef.current.confirm(sessionIdRef.current, yes);
            applyReply(reply);
        }
        catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
        finally {
            setSending(false);
        }
    }, [ensureLoggedIn, applyReply]);
    const reset = (0, react_1.useCallback)(() => {
        setMessages([]);
        setPendingConfirmation(false);
        setConfirmPrompt(null);
        setError(null);
    }, []);
    return { messages, sending, error, pendingConfirmation, confirmPrompt, send, confirm, reset };
}
