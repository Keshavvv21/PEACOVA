# algo-akanade-react-core

The shared chat hook underneath both `<AkanadeChatWidget>` implementations
([`../algo-akanade-react`](../algo-akanade-react) for the web,
[`../algo-akanade-react-native`](../algo-akanade-react-native) for React
Native). Uses only the plain React hooks API
(`useState`/`useCallback`/`useRef`) — no DOM, no React Native
primitives — so the exact same file runs unmodified under React DOM
and React Native, since both share the same underlying `react` package.

## `useAkanadeChat(options)`

```ts
const chat = useAkanadeChat({ baseUrl: 'https://your-host/chatbot', email, password });

chat.messages;            // AkanadeChatMessage[]
chat.sending;              // boolean
chat.error;                // string | null
chat.pendingConfirmation;  // boolean
chat.confirmPrompt;        // string | null
await chat.send('Backtest AAPL for the last 10 years');
await chat.confirm(true);  // answer a pendingConfirmation turn
chat.reset();              // clear history
```

Logs in lazily on the first `send()`/`confirm()` call if you passed
`email`/`password` instead of an existing `accessToken` — every
subsequent call reuses the stored token.

If you don't want either ready-made widget, build your own UI directly
on this hook.

## Tests

```bash
npm install   # from the integrations/ workspace root — see its README
npm test
```
5 tests, using `react-test-renderer` (no jsdom, no real DOM) against a
mocked `fetch`.
