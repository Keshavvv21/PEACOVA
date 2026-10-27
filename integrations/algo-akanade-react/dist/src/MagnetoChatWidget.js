"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.MagnetoChatWidget = MagnetoChatWidget;
const jsx_runtime_1 = require("react/jsx-runtime");
// A complete, drop-in chat widget for any React website. Renders plain
// DOM elements with inline styles only (no CSS framework dependency),
// so it doesn't fight an existing app's own styling. Chat logic comes
// from algo-magneto-react-core's useMagnetoChat (shared with the React
// Native widget); voice comes from algo-magneto-voice's browser
// adapter, and is only offered when the browser actually supports the
// Web Speech API.
const react_1 = require("react");
const algo_magneto_react_core_1 = require("algo-magneto-react-core");
const algo_magneto_voice_1 = require("algo-magneto-voice");
const CHART_IMAGE_RE = /!\[[^\]]*\]\((data:image\/[a-zA-Z]+;base64,[^)]+)\)/;
function isWebSpeechAvailable() {
    if (typeof window === 'undefined')
        return false;
    const w = window;
    return Boolean(w.SpeechRecognition ?? w.webkitSpeechRecognition);
}
function renderMessageText(text) {
    const match = CHART_IMAGE_RE.exec(text);
    if (!match)
        return text;
    const before = text.slice(0, match.index).trim();
    const src = match[1];
    return ((0, jsx_runtime_1.jsxs)(jsx_runtime_1.Fragment, { children: [before && (0, jsx_runtime_1.jsx)("div", { children: before }), (0, jsx_runtime_1.jsx)("img", { src: src, alt: "chart", style: { maxWidth: '100%', marginTop: 8, borderRadius: 4 } })] }));
}
function MagnetoChatWidget(props) {
    const { title = 'Algo AI', placeholder = 'Ask Algo AI…', ...chatOptions } = props;
    const chat = (0, algo_magneto_react_core_1.useMagnetoChat)(chatOptions);
    const [input, setInput] = (0, react_1.useState)('');
    const [listening, setListening] = (0, react_1.useState)(false);
    const voiceAvailable = isWebSpeechAvailable();
    const handleSend = () => {
        const text = input;
        if (!text.trim())
            return;
        setInput('');
        void chat.send(text);
    };
    const handleMic = async () => {
        if (!voiceAvailable || listening)
            return;
        setListening(true);
        try {
            const heard = await (0, algo_magneto_voice_1.createWebSpeechAdapter)().recognizeSpeech();
            setListening(false);
            if (heard.trim())
                await chat.send(heard);
        }
        catch {
            setListening(false);
        }
    };
    return ((0, jsx_runtime_1.jsxs)("div", { style: {
            border: '1px solid #ddd',
            borderRadius: 8,
            width: 360,
            display: 'flex',
            flexDirection: 'column',
            fontFamily: 'system-ui, sans-serif',
            fontSize: 14,
            background: '#fff',
            color: '#111',
        }, children: [(0, jsx_runtime_1.jsx)("div", { style: { padding: 12, fontWeight: 600, borderBottom: '1px solid #eee' }, children: title }), (0, jsx_runtime_1.jsxs)("div", { style: { flex: 1, overflowY: 'auto', padding: 12, maxHeight: 360, minHeight: 120 }, children: [chat.messages.map((m) => ((0, jsx_runtime_1.jsx)("div", { style: { textAlign: m.role === 'user' ? 'right' : 'left', marginBottom: 8 }, children: (0, jsx_runtime_1.jsx)("div", { style: {
                                display: 'inline-block',
                                padding: '6px 10px',
                                borderRadius: 6,
                                maxWidth: '85%',
                                background: m.role === 'user' ? '#1a56db' : '#f3f4f6',
                                color: m.role === 'user' ? '#fff' : '#111',
                                whiteSpace: 'pre-wrap',
                                textAlign: 'left',
                            }, children: renderMessageText(m.text) }) }, m.id))), chat.error && (0, jsx_runtime_1.jsx)("div", { style: { color: '#b91c1c', marginTop: 4 }, children: chat.error })] }), chat.pendingConfirmation && ((0, jsx_runtime_1.jsxs)("div", { style: { padding: 8, borderTop: '1px solid #eee', display: 'flex', alignItems: 'center', gap: 8 }, children: [(0, jsx_runtime_1.jsx)("span", { style: { flex: 1 }, children: chat.confirmPrompt }), (0, jsx_runtime_1.jsx)("button", { type: "button", onClick: () => void chat.confirm(true), disabled: chat.sending, children: "Yes" }), (0, jsx_runtime_1.jsx)("button", { type: "button", onClick: () => void chat.confirm(false), disabled: chat.sending, children: "No" })] })), (0, jsx_runtime_1.jsxs)("div", { style: { display: 'flex', borderTop: '1px solid #eee', padding: 8, gap: 8 }, children: [(0, jsx_runtime_1.jsx)("input", { value: input, onChange: (e) => setInput(e.target.value), onKeyDown: (e) => {
                            if (e.key === 'Enter')
                                handleSend();
                        }, placeholder: placeholder, disabled: chat.sending, style: { flex: 1, minWidth: 0, padding: '6px 8px' } }), voiceAvailable && ((0, jsx_runtime_1.jsx)("button", { type: "button", onClick: () => void handleMic(), disabled: chat.sending || listening, title: "Speak", children: listening ? '…' : '🎤' })), (0, jsx_runtime_1.jsx)("button", { type: "button", onClick: handleSend, disabled: chat.sending || !input.trim(), children: "Send" })] })] }));
}
