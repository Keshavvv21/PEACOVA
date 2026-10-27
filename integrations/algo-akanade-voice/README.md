# algo-akanade-voice

Voice — speech in, speech out — wrapped around
[`@algo-akanade/connector`](../algo-akanade-connector)'s `chat()`, for
a React Native app or a website that wants the same voice experience
the Flutter mobile app already has (`flutter_tts` + on-device speech
recognition). There is **no voice API on the backend** —
`/api/chat/message` is plain text in, text out — so this is entirely
client-side: recognize speech → send the text through the normal chat
call → speak the reply back.

## Pieces

- **`VoiceChatSession`** (`src/voiceChatSession.ts`) — the orchestration:
  `listenAndReply()` and `listenAndConfirm()` (for a `pendingConfirmation`
  turn). Platform-agnostic; takes a `VoiceIO` (`recognizeSpeech()` +
  `speak()`) and a `AkanadeClient`. Fully unit-tested (10 tests) with a
  fake `VoiceIO` — no browser or device needed to run these.
- **`stripForSpeech()`** — strips markdown image syntax (chat replies
  can embed a base64 PNG chart from the Trading Agent's backtests —
  speaking that out loud would be nonsense) and code fences before
  anything reaches a TTS engine.
- **`createWebSpeechAdapter()`** (`src/webAdapters.ts`) — a browser
  `VoiceIO` built on the Web Speech API (`SpeechRecognition` +
  `speechSynthesis`), both real, shipping browser APIs. **Not unit
  tested in this sandbox** (no browser environment here) — it's
  standard usage of those two APIs, not exploratory code, but has not
  been run against a real browser.
- **`createReactNativeVoiceAdapter()`** (`src/reactNativeAdapters.ts`) —
  a `VoiceIO` built on `@react-native-voice/voice` and `react-native-tts`
  (the two most common RN voice libraries). Neither library is a
  dependency of this package — pass your app's own imported modules in
  — so this stays installable without React Native itself. The wiring
  against those libraries' documented event shapes **is** unit-tested
  (4 tests, fakes matching their interfaces), but has not run against
  the real native modules on a device.

## Usage

### Website (Web Speech API)

```ts
import { AkanadeClient } from '@algo-akanade/connector';
import { VoiceChatSession, createWebSpeechAdapter } from 'algo-akanade-voice';

const akanade = new AkanadeClient({ baseUrl: 'https://your-host/chatbot' });
await akanade.login('you@company.com', 'your-password');

const session = new VoiceChatSession({
  akanade,
  sessionId: 'voice-session-1',
  voice: createWebSpeechAdapter(),
});

button.onclick = async () => {
  const { heard, spoken } = await session.listenAndReply();
  console.log('You said:', heard, '— Algo AI said:', spoken);
};
```

### React Native

```ts
import Voice from '@react-native-voice/voice';
import Tts from 'react-native-tts';
import { AkanadeClient } from '@algo-akanade/connector';
import { VoiceChatSession, createReactNativeVoiceAdapter } from 'algo-akanade-voice';

const akanade = new AkanadeClient({ baseUrl: 'https://your-host/chatbot' });
await akanade.login('you@company.com', 'your-password');

const session = new VoiceChatSession({
  akanade,
  sessionId: 'voice-session-1',
  voice: createReactNativeVoiceAdapter({ Voice, Tts }),
});

const { heard, spoken } = await session.listenAndReply();
```

## Tests

```bash
npm test
```
10 tests, all passing, no browser/device/live backend required.
