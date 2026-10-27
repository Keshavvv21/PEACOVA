export interface LoginResult {
    accessToken: string;
}
export interface ChatReply {
    reply: string;
    pendingConfirmation: boolean;
    confirmPrompt: string | null;
    provider: string | null;
}
export interface TrainAlgoAiResult {
    domain: string;
    topic: string;
    source: string;
    charsLogged: number;
    ragChunksIndexed: number;
    jobStatusNote: string | null;
}
export interface RunningTrainingJobs {
    jobs: Array<Record<string, unknown>>;
    running: string[];
}
