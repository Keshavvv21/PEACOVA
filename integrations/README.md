# Integrations

Packages that connect existing, third-party software to ALGO AKANADE's
existing REST API (documented in
[`README_EXTERNAL_API.md`](../orchestrator_framework_layer/akanade/README_EXTERNAL_API.md)).
No new backend endpoints were added to build any of these — they all
wrap the same `/api/auth/login` and `/api/chat/message` calls the web
app, mobile app, and VS Code extension already make.

**Want to add this to your own website/React Native app/Flutter
app/Tally right now?** See [`HOW_TO_PLUG_IN.md`](HOW_TO_PLUG_IN.md) —
this file is the architecture; that one is the copy-pasteable how-to,
including a table of exactly which platforms have voice and a
ready-made widget today.

## How it fits together

```
                         ALGO AKANADE backend
                    /api/auth/login, /api/chat/message,
                    /api/peft-training/*, /api/admin/peft-jobs
                                  ▲
                                  │  plain HTTPS/JSON
                                  │
              ┌───────────────────┴───────────────────┐
              │                                        │
    algo-akanade-connector (TS)              algo_akanade_connector_dart
    used by Node / any JS backend             (for a Flutter app other than
                    ▲                          tool_layer_akanade/mobile,
                    │                           which has its own client)
       ┌────────────┼─────────────┐                    ▲
       │             │             │                    │
algo-akanade-voice   │      tally-connector    algo_akanade_voice_dart
(speech in/out,      │      (Tally's XML gateway (speech_to_text +
 wraps chat() —      │       -> chat()/Train      flutter_tts, wraps
 Web Speech API/     │        Algo AI)            chat() — for Flutter)
 RN voice libs)      │
       ▲             │
       │             │
algo-akanade-react-core (useAkanadeChat — plain React hooks,
runs unmodified in both React DOM and React Native)
       ▲                                  ▲
       │                                  │
algo-akanade-react                algo-akanade-react-native
<AkanadeChatWidget> for            <AkanadeChatWidget> for
any React website                  any React Native app
(+ voice via the browser           (+ voice via
 Web Speech API)                    @react-native-voice/voice
                                     + react-native-tts)
```

**The idea**: any existing software — a website, a React Native app, a
Flutter app, or Tally — gets ALGO AKANADE's chat/Trading-Agent/Train-Algo-AI
functionality by importing one small package and pointing it at your
deployed backend's URL and credentials. Nothing here is a separate
product; it's all the same REST surface, wrapped once per platform so
you don't hand-roll `fetch`/`http` calls yourself.

## The packages

- **[`algo-akanade-connector`](algo-akanade-connector)** — the plain
  TypeScript client (login, chat, Train Algo AI, running-jobs) for
  Node, React, or React Native directly. Not a branded "SDK" — just
  the handful of functions any external caller needs.
- **[`algo_akanade_connector_dart`](algo_akanade_connector_dart)** —
  the same client, in Dart, for any Flutter app other than
  `tool_layer_akanade/mobile` (ALGO AKANADE's own official app, which has
  its own separate client already).
- **[`algo-akanade-voice`](algo-akanade-voice)** — speech-to-text in,
  text-to-speech out, wrapped around the TypeScript connector's
  `chat()`, for web (browser Web Speech API) and React Native
  (`@react-native-voice/voice` + `react-native-tts`). There is **no
  voice API on the backend at all** — `/api/chat/message` is plain
  text in, text out — so this is entirely client-side orchestration,
  the same pattern the Flutter mobile app already uses with
  `flutter_tts`.
- **[`algo_akanade_voice_dart`](algo_akanade_voice_dart)** — the Dart
  twin, for a Flutter app, built on `speech_to_text` + `flutter_tts`
  (both dependency-injected, not hard dependencies of this package —
  see its README for the two-class wrapper a real app needs).
- **[`algo-akanade-react-core`](algo-akanade-react-core)** — the
  `useAkanadeChat()` hook: message history, `send()`, `confirm()`,
  loading/error state. Pure React hooks, no DOM or RN-specific code —
  the identical file runs under both React DOM and React Native.
- **[`algo-akanade-react`](algo-akanade-react)** — a ready-made
  `<AkanadeChatWidget>` **for any React website**: drop it in with a
  `baseUrl` and credentials and it renders a working chat UI, complete
  with inline Trading-Agent chart images and a mic button when the
  browser supports it. Plain inline styles, so it won't clash with an
  existing app's own CSS/design system.
- **[`algo-akanade-react-native`](algo-akanade-react-native)** — the
  same `<AkanadeChatWidget>`, **for any React Native app**, built on
  the identical `useAkanadeChat()` hook with RN's own core components
  instead of DOM elements. Voice is optional — pass your app's own
  `@react-native-voice/voice` + `react-native-tts` modules to enable
  the mic button.
- **[`tally-connector`](tally-connector)** — a concrete, working Tally
  plugin, packaged as a single downloadable program (`npm run
  package:all` builds a Windows `.exe` and a Linux binary, no Node.js
  install required to run them). A non-technical user downloads it,
  double-clicks it, and fills in a one-time setup form in their
  browser — it then serves its own popup chat window (with voice, via
  the same browser Web Speech API the web widget uses) with one-click
  buttons to analyze ledgers or the Day Book, exported live from
  Tally's own XML HTTP gateway. An optional TDL snippet adds an "Ask
  Algo AI" button inside Tally's own menu too, but the popup chat works
  standalone without it.

There is no ready-made `<AkanadeChatWidget>` for Flutter yet — only the
connector and voice pieces — see `HOW_TO_PLUG_IN.md` for how to wire
them into your own screen in the meantime.

## Running the tests

The TypeScript packages are an **npm workspace** rooted here — this is
what keeps `react` a single, shared copy across
`algo-akanade-react-core`/`algo-akanade-react` (two separate copies of
React break hooks with a real "Invalid hook call" error, which is
exactly what happened before this was set up as a workspace):

```bash
cd integrations
npm install               # once, hoists shared deps for every workspace package
npm test --workspace=algo-akanade-connector
npm test --workspace=algo-akanade-voice
npm test --workspace=algo-akanade-react-core
npm test --workspace=algo-akanade-react
npm test --workspace=tally-connector
```

(Each package's own directory also still works standalone —
`cd algo-akanade-react && npm test` — as long as `npm install` has
been run from the workspace root first.)

`algo-akanade-react-native` ships raw `.tsx` source with no build step
(Metro transpiles it directly) and is not part of the npm workspace's
test run — see its README's honesty note.

The two Dart packages are separate from the npm workspace:

```bash
cd algo_akanade_connector_dart && dart pub get && dart test
cd ../algo_akanade_voice_dart && dart pub get && dart test
```

**71 tests total** across every package — 7 + 10 + 5 + 5 + 27 in the
npm workspace, plus 7 + 10 in Dart — all against mocked/fake HTTP and
Tally fixtures. No live backend, browser, device, or Tally instance required
to run any of them.
