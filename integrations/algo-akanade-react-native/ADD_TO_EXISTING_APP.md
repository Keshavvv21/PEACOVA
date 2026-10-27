# Adding full chat + voice to an already-existing React Native app

You have a React Native app already shipping, and you want to drop in
a working Algo AI chat screen — text and voice — without restructuring
anything you already have. This is the step-by-step for that.

## Step 1 — Add the packages

```bash
# From your existing app's root:
npm install algo-akanade-react-core algo-akanade-voice
# or, using the local copies in this repo:
# "algo-akanade-react-core": "file:../path/to/integrations/algo-akanade-react-core"
# "algo-akanade-voice": "file:../path/to/integrations/algo-akanade-voice"
# "algo-akanade-react-native": "file:../path/to/integrations/algo-akanade-react-native"
```

Since your app already has `react` and `react-native` installed, no
version conflicts to worry about — these packages declare them as
peer dependencies, not their own copies.

## Step 2 — Voice libraries (skip this if you only want text chat)

```bash
npm install @react-native-voice/voice react-native-tts
```

### iOS — add usage descriptions and run pod install

In `ios/YourApp/Info.plist`, inside the top-level `<dict>`:

```xml
<key>NSMicrophoneUsageDescription</key>
<string>Used so you can talk to Algo AI.</string>
<key>NSSpeechRecognitionUsageDescription</key>
<string>Used to understand what you say to Algo AI.</string>
```

Then:

```bash
cd ios && pod install && cd ..
```

### Android — add the microphone permission

In `android/app/src/main/AndroidManifest.xml`, alongside your existing
`<uses-permission>` entries:

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

Android 6+ also needs the **runtime** permission requested once before
first use — a one-time addition wherever your app already does this
kind of thing (or right before showing the chat screen):

```ts
import { PermissionsAndroid, Platform } from 'react-native';

async function ensureMicPermission() {
  if (Platform.OS !== 'android') return true;
  const granted = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.RECORD_AUDIO);
  return granted === PermissionsAndroid.RESULTS.GRANTED;
}
```

## Step 3 — Add the chat screen

Wherever your app's navigation is set up (React Navigation, a custom
switch, a modal — this is a plain component, it doesn't care), add a
new screen:

```tsx
// AlgoAiChatScreen.tsx
import Voice from '@react-native-voice/voice';
import Tts from 'react-native-tts';
import { AkanadeChatWidget } from 'algo-akanade-react-native';
import { useAuth } from './your-existing-auth-hook'; // however your app already tracks login

export function AlgoAiChatScreen() {
  const { accessToken } = useAuth(); // or hardcode email/password for a quick test

  return (
    <AkanadeChatWidget
      baseUrl="https://your-host/chatbot"
      accessToken={accessToken}
      voiceModules={{ Voice, Tts }}
    />
  );
}
```

Register it the same way you'd register any other screen:

```tsx
// React Navigation example — adapt to whatever your app already uses
<Stack.Screen name="AlgoAI" component={AlgoAiChatScreen} options={{ title: 'Algo AI' }} />
```

If your app already has its own auth/token storage, pass that token in
as `accessToken` — no need to have `AkanadeChatWidget` log in itself.
If it doesn't, pass `email`/`password` directly and the widget logs in
lazily on first message.

## Step 4 — Test it

1. Text: open the screen, type a message, confirm you get a real reply
   from your ALGO AKANADE deployment.
2. Voice: tap the mic button (only shown because `voiceModules` was
   passed), grant the microphone permission when prompted (first time
   only), speak, confirm it's transcribed and sent, and confirm the
   reply is spoken back (charts/code blocks are stripped from speech
   automatically — see `algo-akanade-voice`'s `stripForSpeech`).
3. Try a PM Tool prompt that needs confirmation (e.g. "delete my task
   X") and confirm the Yes/No buttons appear and both paths work.

## What you get vs. what you keep building yourself

**Included**: message history, sending, the confirm flow, chart
rendering for Trading Agent backtests, the mic button and its
recognize→send→speak loop.

**Not included** (build these the way you'd build them for any other
screen in your app): persisting chat history across app restarts,
push notifications for async replies, a custom look beyond
`AkanadeChatWidget`'s built-in RN styles (if you need different
styling, use `useAkanadeChat` from `algo-akanade-react-core` directly
and render your own UI on top of it — the widget is a convenience, not
the only way in).

## Honesty note

`AkanadeChatWidget` (this package) has not been built or run against a
real `react-native` install in this session — there is no React
Native/device/simulator toolchain available in the sandbox this was
built in. It's correct, standard usage of RN's core components and the
two named voice libraries' documented APIs. The chat logic underneath
it (`useAkanadeChat`, from `algo-akanade-react-core`) **is** unit
tested (5 tests) since that file has no RN-specific code at all — walk
through Step 4 above in your own app to confirm the RN-specific parts
end-to-end.
