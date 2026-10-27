export interface BridgeConfig {
    magnetoBaseUrl: string;
    magnetoEmail: string;
    magnetoPassword: string;
    tallyBaseUrl: string;
    bridgePort: number;
}
export declare const DEFAULT_TALLY_BASE_URL = "http://localhost:9000";
export declare const DEFAULT_BRIDGE_PORT = 4600;
export declare function configFilePath(baseDir: string): string;
/** Returns null if no config has been saved yet, or the file is unreadable/incomplete. */
export declare function loadConfig(baseDir: string): BridgeConfig | null;
export declare function saveConfig(baseDir: string, config: BridgeConfig): void;
/** Dev/CI convenience only — lets the old environment-variable flow keep working for anyone already using it; a real end user never sets these. */
export declare function loadConfigFromEnv(): BridgeConfig | null;
