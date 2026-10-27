"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.VoiceChatSession = void 0;
exports.stripForSpeech = stripForSpeech;
/**
 * Chat replies can embed a base64 PNG chart (the Trading Agent's
 * backtest charts — see trading_tools.py's chart_base64) or fenced
 * code blocks. Neither is speakable, so this strips them before
 * anything is passed to a text-to-speech engine.
 */
function stripForSpeech(text) {
    return text
        .replace(/!\[[^\]]*\]\(data:[^)]+\)/g, '')
        .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
        .replace(/```[\s\S]*?```/g, ' (code block omitted) ')
        .replace(/`([^`]+)`/g, '$1')
        .replace(/\s+/g, ' ')
        .trim();
}
class VoiceChatSession {
    constructor(options) {
        this.shiki = options.shiki;
        this.sessionId = options.sessionId;
        this.voice = options.voice;
        this.provider = options.provider ?? null;
    }
    /** Listens for one spoken turn, sends it to ALGO SHIKI, and speaks the reply back. */
    async listenAndReply() {
        const heard = await this.voice.recognizeSpeech();
        const reply = await this.shiki.chat(heard, this.sessionId, this.provider);
        const spoken = stripForSpeech(reply.reply);
        if (spoken)
            await this.voice.speak(spoken);
        return { heard, reply, spoken };
    }
    /** For a pendingConfirmation turn — listens for a yes/no-shaped utterance and sends it as the confirmation. */
    async listenAndConfirm() {
        const heard = await this.voice.recognizeSpeech();
        const yes = /^\s*(yes|yeah|yep|confirm|sure|go ahead)\b/i.test(heard);
        const reply = await this.shiki.confirm(this.sessionId, yes);
        const spoken = stripForSpeech(reply.reply);
        if (spoken)
            await this.voice.speak(spoken);
        return { heard, reply, spoken };
    }
}
exports.VoiceChatSession = VoiceChatSession;
