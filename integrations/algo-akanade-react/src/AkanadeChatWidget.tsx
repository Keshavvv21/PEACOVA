// A complete, drop-in chat widget for any React website. Renders plain
// DOM elements with inline styles only (no CSS framework dependency),
// so it doesn't fight an existing app's own styling. Chat logic comes
// from algo-akanade-react-core's useAkanadeChat (shared with the React
// Native widget); voice comes from algo-akanade-voice's browser
// adapter, and is only offered when the browser actually supports the
// Web Speech API.
import { useState, type JSX, type ReactNode } from 'react';
import { useAkanadeChat, type UseAkanadeChatOptions } from 'algo-akanade-react-core';
import { createWebSpeechAdapter } from 'algo-akanade-voice';

export interface AkanadeChatWidgetProps extends UseAkanadeChatOptions {
  title?: string;
  placeholder?: string;
}

const CHART_IMAGE_RE = /!\[[^\]]*\]\((data:image\/[a-zA-Z]+;base64,[^)]+)\)/;

function isWebSpeechAvailable(): boolean {
  if (typeof window === 'undefined') return false;
  const w = window as unknown as Record<string, unknown>;
  return Boolean(w.SpeechRecognition ?? w.webkitSpeechRecognition);
}

function renderMessageText(text: string): ReactNode {
  const match = CHART_IMAGE_RE.exec(text);
  if (!match) return text;
  const before = text.slice(0, match.index).trim();
  const src = match[1];
  return (
    <>
      {before && <div>{before}</div>}
      {/* eslint-disable-next-line @next/next/no-img-element -- plain img keeps this framework-agnostic */}
      <img src={src} alt="chart" style={{ maxWidth: '100%', marginTop: 8, borderRadius: 4 }} />
    </>
  );
}

export function AkanadeChatWidget(props: AkanadeChatWidgetProps): JSX.Element {
  const { title = 'Algo AI', placeholder = 'Ask Algo AI…', ...chatOptions } = props;
  const chat = useAkanadeChat(chatOptions);
  const [input, setInput] = useState('');
  const [listening, setListening] = useState(false);
  const voiceAvailable = isWebSpeechAvailable();

  const handleSend = (): void => {
    const text = input;
    if (!text.trim()) return;
    setInput('');
    void chat.send(text);
  };

  const handleMic = async (): Promise<void> => {
    if (!voiceAvailable || listening) return;
    setListening(true);
    try {
      const heard = await createWebSpeechAdapter().recognizeSpeech();
      setListening(false);
      if (heard.trim()) await chat.send(heard);
    } catch {
      setListening(false);
    }
  };

  return (
    <div
      style={{
        border: '1px solid #ddd',
        borderRadius: 8,
        width: 360,
        display: 'flex',
        flexDirection: 'column',
        fontFamily: 'system-ui, sans-serif',
        fontSize: 14,
        background: '#fff',
        color: '#111',
      }}
    >
      <div style={{ padding: 12, fontWeight: 600, borderBottom: '1px solid #eee' }}>{title}</div>

      <div style={{ flex: 1, overflowY: 'auto', padding: 12, maxHeight: 360, minHeight: 120 }}>
        {chat.messages.map((m) => (
          <div key={m.id} style={{ textAlign: m.role === 'user' ? 'right' : 'left', marginBottom: 8 }}>
            <div
              style={{
                display: 'inline-block',
                padding: '6px 10px',
                borderRadius: 6,
                maxWidth: '85%',
                background: m.role === 'user' ? '#1a56db' : '#f3f4f6',
                color: m.role === 'user' ? '#fff' : '#111',
                whiteSpace: 'pre-wrap',
                textAlign: 'left',
              }}
            >
              {renderMessageText(m.text)}
            </div>
          </div>
        ))}
        {chat.error && <div style={{ color: '#b91c1c', marginTop: 4 }}>{chat.error}</div>}
      </div>

      {chat.pendingConfirmation && (
        <div style={{ padding: 8, borderTop: '1px solid #eee', display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ flex: 1 }}>{chat.confirmPrompt}</span>
          <button type="button" onClick={() => void chat.confirm(true)} disabled={chat.sending}>
            Yes
          </button>
          <button type="button" onClick={() => void chat.confirm(false)} disabled={chat.sending}>
            No
          </button>
        </div>
      )}

      <div style={{ display: 'flex', borderTop: '1px solid #eee', padding: 8, gap: 8 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSend();
          }}
          placeholder={placeholder}
          disabled={chat.sending}
          style={{ flex: 1, minWidth: 0, padding: '6px 8px' }}
        />
        {voiceAvailable && (
          <button type="button" onClick={() => void handleMic()} disabled={chat.sending || listening} title="Speak">
            {listening ? '…' : '🎤'}
          </button>
        )}
        <button type="button" onClick={handleSend} disabled={chat.sending || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}
