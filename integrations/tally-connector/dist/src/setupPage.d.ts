import type { BridgeConfig } from './config';
export declare function renderSetupPage(existing?: Partial<BridgeConfig>, opts?: {
    saved?: boolean;
    error?: string;
}): string;
