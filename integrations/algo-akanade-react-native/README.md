# algo-akanade-react-native

A ready-made `<AkanadeChatWidget>` for any React Native app — chat plus
optional voice, built on the same `useAkanadeChat` hook the web widget
([`../algo-akanade-react`](../algo-akanade-react)) uses.

**Adding this to an app you already have shipping?** See
[`ADD_TO_EXISTING_APP.md`](ADD_TO_EXISTING_APP.md) for the full
step-by-step — package install, the native permission setup voice
needs on iOS/Android, wiring the screen into your existing navigation,
and how to test it end-to-end.

## Install

```yaml
# package.json
"dependencies": {
  "algo-akanade-react-native": "file:../path/to/integrations/algo-akanade-react-native"
}
```

Ships as raw `.tsx` source, no build step — Metro (React Native's own
bundler) transpiles local/workspace TypeScript packages the same way
it transpiles your app's own code, so there's nothing to compile here.

## Usage

```tsx
import { AkanadeChatWidget } from 'algo-akanade-react-native';

<AkanadeChatWidget
  baseUrl="https://your-host/chatbot"
  email="you@company.com"
  password="your-password"
/>
```

### With voice

Pass your app's own already-imported `@react-native-voice/voice` and
`react-native-tts` modules (both need native linking — a normal RN
install step, not something this package does for you):

```tsx
import Voice from '@react-native-voice/voice';
import Tts from 'react-native-tts';
import { AkanadeChatWidget } from 'algo-akanade-react-native';

<AkanadeChatWidget
  baseUrl="https://your-host/chatbot"
  accessToken={myStoredToken}
  voiceModules={{ Voice, Tts }}
/>
```

Omit `voiceModules` and the widget is chat-only (no mic button shown).

## Honesty note

This file has not been built or run in this session — there is no
`react-native` install, simulator, or device available in the
environment this was built in. It is real, correct usage of React
Native's own core components (`View`/`Text`/`TextInput`/
`TouchableOpacity`/`Image`/`ScrollView`) and the two voice libraries'
documented APIs, not exploratory code — but verify it inside your own
RN project before shipping. The chat logic underneath it
(`useAkanadeChat`, from `algo-akanade-react-core`) **is** unit tested
(5 tests, via `react-test-renderer`) since that file has no
RN-specific code in it at all — the same hook is what the web widget
uses too.
