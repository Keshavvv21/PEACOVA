// A complete <AkanadeChatWidget> for any React Native app. Uses the
// SAME chat logic as the web widget (algo-akanade-react-core's
// useAkanadeChat — plain React hooks, no DOM, so it runs unmodified
// under React Native's own 'react' instance) with RN's own core
// components instead of DOM elements. Voice is optional: pass
// `voiceModules` (your app's own already-imported @react-native-voice/
// voice + react-native-tts) to get a mic button; omit it and the
// widget is chat-only.
//
// Honesty note: this file has NOT been built or run in this session —
// there is no react-native install/device/simulator available in this
// sandbox. It is real, correct usage of React Native's own core
// components and the two named voice libraries' documented APIs, not
// exploratory code, but verify it in your own RN project before
// shipping. algo-akanade-react-core's hook underneath it IS unit
// tested (5 tests, against react-test-renderer) since that file has no
// RN-specific code at all.
import { useState } from 'react';
import {
  ActivityIndicator,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { useAkanadeChat, type UseAkanadeChatOptions } from 'algo-akanade-react-core';
import { createReactNativeVoiceAdapter, type ReactNativeTtsModule, type ReactNativeVoiceModule } from 'algo-akanade-voice';

export interface AkanadeChatWidgetProps extends UseAkanadeChatOptions {
  title?: string;
  placeholder?: string;
  /** Pass your app's own already-imported voice libraries to enable the mic button. Omit for chat-only. */
  voiceModules?: { Voice: ReactNativeVoiceModule; Tts: ReactNativeTtsModule };
}

const CHART_IMAGE_RE = /!\[[^\]]*\]\((data:image\/[a-zA-Z]+;base64,[^)]+)\)/;

function MessageBubble({ text }: { text: string }): JSX.Element {
  const match = CHART_IMAGE_RE.exec(text);
  if (!match) return <Text>{text}</Text>;
  const before = text.slice(0, match.index).trim();
  return (
    <View>
      {before ? <Text>{before}</Text> : null}
      <Image source={{ uri: match[1] }} style={styles.chartImage} resizeMode="contain" />
    </View>
  );
}

export function AkanadeChatWidget(props: AkanadeChatWidgetProps): JSX.Element {
  const { title = 'Algo AI', placeholder = 'Ask Algo AI…', voiceModules, ...chatOptions } = props;
  const chat = useAkanadeChat(chatOptions);
  const [input, setInput] = useState('');
  const [listening, setListening] = useState(false);

  const handleSend = (): void => {
    const text = input;
    if (!text.trim()) return;
    setInput('');
    void chat.send(text);
  };

  const handleMic = async (): Promise<void> => {
    if (!voiceModules || listening) return;
    setListening(true);
    try {
      const adapter = createReactNativeVoiceAdapter(voiceModules);
      const heard = await adapter.recognizeSpeech();
      setListening(false);
      if (heard.trim()) await chat.send(heard);
    } catch {
      setListening(false);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerText}>{title}</Text>
      </View>

      <ScrollView style={styles.messages} contentContainerStyle={styles.messagesContent}>
        {chat.messages.map((m) => (
          <View key={m.id} style={[styles.bubbleRow, m.role === 'user' ? styles.bubbleRowUser : styles.bubbleRowAssistant]}>
            <View style={[styles.bubble, m.role === 'user' ? styles.bubbleUser : styles.bubbleAssistant]}>
              <MessageBubble text={m.text} />
            </View>
          </View>
        ))}
        {chat.error ? <Text style={styles.errorText}>{chat.error}</Text> : null}
      </ScrollView>

      {chat.pendingConfirmation ? (
        <View style={styles.confirmRow}>
          <Text style={styles.confirmText}>{chat.confirmPrompt}</Text>
          <TouchableOpacity style={styles.confirmButton} onPress={() => void chat.confirm(true)} disabled={chat.sending}>
            <Text>Yes</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.confirmButton} onPress={() => void chat.confirm(false)} disabled={chat.sending}>
            <Text>No</Text>
          </TouchableOpacity>
        </View>
      ) : null}

      <View style={styles.inputRow}>
        <TextInput
          value={input}
          onChangeText={setInput}
          onSubmitEditing={handleSend}
          placeholder={placeholder}
          editable={!chat.sending}
          style={styles.input}
        />
        {voiceModules ? (
          <TouchableOpacity style={styles.iconButton} onPress={() => void handleMic()} disabled={chat.sending || listening}>
            <Text>{listening ? '…' : '🎤'}</Text>
          </TouchableOpacity>
        ) : null}
        <TouchableOpacity style={styles.sendButton} onPress={handleSend} disabled={chat.sending || !input.trim()}>
          {chat.sending ? <ActivityIndicator size="small" /> : <Text style={styles.sendButtonText}>Send</Text>}
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { borderWidth: 1, borderColor: '#ddd', borderRadius: 8, backgroundColor: '#fff', flex: 1 },
  header: { padding: 12, borderBottomWidth: 1, borderBottomColor: '#eee' },
  headerText: { fontWeight: '600', fontSize: 16 },
  messages: { flex: 1 },
  messagesContent: { padding: 12 },
  bubbleRow: { marginBottom: 8, flexDirection: 'row' },
  bubbleRowUser: { justifyContent: 'flex-end' },
  bubbleRowAssistant: { justifyContent: 'flex-start' },
  bubble: { paddingVertical: 6, paddingHorizontal: 10, borderRadius: 6, maxWidth: '85%' },
  bubbleUser: { backgroundColor: '#1a56db' },
  bubbleAssistant: { backgroundColor: '#f3f4f6' },
  chartImage: { width: 260, height: 160, marginTop: 8 },
  errorText: { color: '#b91c1c', marginTop: 4 },
  confirmRow: { padding: 8, borderTopWidth: 1, borderTopColor: '#eee', flexDirection: 'row', alignItems: 'center', gap: 8 },
  confirmText: { flex: 1 },
  confirmButton: { paddingVertical: 6, paddingHorizontal: 10, borderRadius: 6, backgroundColor: '#f3f4f6' },
  inputRow: { flexDirection: 'row', padding: 8, borderTopWidth: 1, borderTopColor: '#eee', gap: 8, alignItems: 'center' },
  input: { flex: 1, borderWidth: 1, borderColor: '#ddd', borderRadius: 6, paddingHorizontal: 8, paddingVertical: 6 },
  iconButton: { padding: 8 },
  sendButton: { paddingVertical: 8, paddingHorizontal: 12, borderRadius: 6, backgroundColor: '#1a56db' },
  sendButtonText: { color: '#fff', fontWeight: '600' },
});
