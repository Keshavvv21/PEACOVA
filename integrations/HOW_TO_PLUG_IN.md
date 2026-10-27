# Adding Algo AI to your own software

You have a website, a React Native app, a Flutter app, or Tally, and
you want it to be able to chat with ALGO AKANADE (PM Tool, Trading Agent,
Train Algo AI) — with voice, where that makes sense. This is the
practical "how do I plug this in" guide; the architecture diagram and
per-package details are in [`README.md`](README.md).

**The short version**: every platform talks to the same two backend
calls (`POST /api/auth/login`, `POST /api/chat/message`) — nothing
here is a separate product per platform, just one wrapper per runtime
so you don't hand-roll HTTP calls yourself.

## Does voice work everywhere?

Yes, everywhere there's a real chat window for a human to talk to —
including Tally now, via its own popup chat window (a real browser
tab the bridge program opens, not literally embedded inside Tally's
window — Tally has no such embedding capability).

| Your software | Connector (talks to ALGO AKANADE) | Voice (mic + speech) | Ready-made chat UI |
|---|---|---|---|
| A website (React) | `algo-akanade-connector` | `algo-akanade-voice` (browser Web Speech API) | `algo-akanade-react`'s `<AkanadeChatWidget>` |
| A React Native app | `algo-akanade-connector` | `algo-akanade-voice` (`@react-native-voice/voice` + `react-native-tts`) | `algo-akanade-react-native`'s `<AkanadeChatWidget>` |
| A Flutter app | `algo_akanade_connector_dart` | `algo_akanade_voice_dart` (`speech_to_text` + `flutter_tts`) | not built yet — wire `VoiceChatSession`/`AkanadeClient` into your own screen (see below) |
| Tally | `tally-connector` | Browser Web Speech API, same as the website widget — the popup chat is a plain HTML page | A popup chat window the bridge program itself serves (`tally-connector/src/chatPage.ts`), plus an optional Tally-menu button via TDL |
| Any other software (a Node backend, another ERP's scripting layer, a plain script) | `algo-akanade-connector`, or just call the REST API directly | only if that software has its own UI with a mic | build your own on top of the connector |

## Website (React)

```bash
npm install --save algo-akanade-react
# (or, if you copied the packages locally: "algo-akanade-react": "file:../algo-akanade-react")
```

```tsx
import { AkanadeChatWidget } from 'algo-akanade-react';

function App() {
  return (
    <AkanadeChatWidget
      baseUrl="https://your-host/chatbot"
      email="you@company.com"
      password="your-password"
    />
  );
}
```

That's it — a working chat widget with a mic button (shown
automatically if the visitor's browser supports it) and inline chart
rendering for Trading Agent backtests. No CSS to configure; it renders
with its own inline styles so it won't clash with your app's design
system.

## React Native app

**Adding this to an app you already have shipping?** See
[`algo-akanade-react-native/ADD_TO_EXISTING_APP.md`](algo-akanade-react-native/ADD_TO_EXISTING_APP.md)
for the full walkthrough, including the native permission setup voice
needs on iOS/Android.

```tsx
import { AkanadeChatWidget } from 'algo-akanade-react-native';

<AkanadeChatWidget
  baseUrl="https://your-host/chatbot"
  email="you@company.com"
  password="your-password"
/>
```

For voice, install `@react-native-voice/voice` and `react-native-tts`
in your own app (they need native linking — a normal RN install step)
and pass them in:

```tsx
import Voice from '@react-native-voice/voice';
import Tts from 'react-native-tts';

<AkanadeChatWidget
  baseUrl="https://your-host/chatbot"
  accessToken={myStoredToken}
  voiceModules={{ Voice, Tts }}
/>
```

## Flutter app

There's no ready-made widget yet, but the pieces are all there —
wiring them into your own screen is a small amount of code:

```dart
import 'package:algo_akanade_connector/algo_akanade_connector.dart';
import 'package:algo_akanade_voice/algo_akanade_voice.dart';

final akanade = AkanadeClient(baseUrl: 'https://your-host/chatbot');
await akanade.login('you@company.com', 'your-password');

// Text chat:
final reply = await akanade.chat('Backtest AAPL for the last 10 years', 'session-1');
print(reply.reply);

// Voice (see algo_akanade_voice's README for the two small wrapper
// classes speech_to_text/flutter_tts need):
final session = VoiceChatSession(
  akanade: akanade,
  sessionId: 'voice-session-1',
  voice: FlutterVoiceAdapter(speechToText: myRealSpeechToText, tts: myRealFlutterTts),
);
final result = await session.listenAndReply();
```

## Tally

**Not a software person, and already have Tally installed?** See
[`tally-connector/CONNECT_YOUR_TALLY.md`](tally-connector/CONNECT_YOUR_TALLY.md)
— download one program, double-click it, fill in one form in your
browser. It opens a real popup chat window (with voice, and two
one-click buttons to analyze your ledgers or Day Book), plus an
optional "Ask Algo AI" button inside Tally's own menu if you also load
the included TDL file — but that's optional, the popup chat works on
its own with no TDL editing at all.

## Anything else

If your software isn't one of the above, you don't need any package
here at all — call the two endpoints directly (documented in full,
with a curl example, in
[`README_EXTERNAL_API.md`](../orchestrator_framework_layer/akanade/README_EXTERNAL_API.md)):

```
POST /api/auth/login        -> { access_token }
POST /api/chat/message      -> { reply, pending_confirmation, confirm_prompt, provider }
```

`algo-akanade-connector` (or its Dart twin) is just a thin, tested
wrapper around exactly those two calls plus Train Algo AI — copy its
`src/index.ts` logic directly into your own codebase if you'd rather
not add a dependency at all.
