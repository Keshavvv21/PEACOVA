# algo-akanade-react

A ready-made `<AkanadeChatWidget>` for any React website: chat, an
inline chart image for Trading Agent backtests, and voice (a mic
button — shown only when the browser actually supports the Web Speech
API). Plain inline styles, no CSS framework dependency, so it drops
into an existing app without fighting its styling.

## Install

```json
"dependencies": {
  "algo-akanade-react": "file:../path/to/integrations/algo-akanade-react"
}
```

## Usage

```tsx
import { AkanadeChatWidget } from 'algo-akanade-react';

<AkanadeChatWidget
  baseUrl="https://your-host/chatbot"
  email="you@company.com"
  password="your-password"
/>
```

Or with an existing token and a fixed session:

```tsx
<AkanadeChatWidget baseUrl="https://your-host/chatbot" accessToken={token} sessionId="my-app-session" />
```

Prefer to build your own UI? Import `useAkanadeChat` from
`algo-akanade-react-core` directly instead of the widget.

## How the pieces fit

```
AkanadeChatWidget (this package)
   ├── useAkanadeChat()        — algo-akanade-react-core (shared with the RN widget)
   │      └── AkanadeClient    — @algo-akanade/connector (the plain REST client)
   └── createWebSpeechAdapter() — algo-akanade-voice (browser mic/TTS)
```

No new backend endpoints — this is UI wrapped around the same
`/api/auth/login` / `/api/chat/message` calls documented in
`README_EXTERNAL_API.md`.

## Tests

```bash
npm install   # from the integrations/ workspace root — see its README
npm test
```
5 tests, using `react-test-renderer` (drives the rendered `<input>`/
`<button>` elements' props directly — no jsdom, no real browser) against
a mocked `fetch`.
