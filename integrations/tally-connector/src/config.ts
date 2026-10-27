// A local config file next to the running program — no environment
// variables, no command line flags, so a non-technical user never
// needs a terminal after the very first setup page. Read/written as
// plain JSON so it's easy to find and delete by hand if needed.
import fs from 'node:fs';
import path from 'node:path';

export interface BridgeConfig {
  algoAkanadeBaseUrl: string;
  algoAkanadeEmail: string;
  algoAkanadePassword: string;
  tallyBaseUrl: string;
  bridgePort: number;
}

export const DEFAULT_TALLY_BASE_URL = 'http://localhost:9000';
export const DEFAULT_BRIDGE_PORT = 4600;
const CONFIG_FILE_NAME = 'algo-tally-config.json';

export function configFilePath(baseDir: string): string {
  return path.join(baseDir, CONFIG_FILE_NAME);
}

/** Returns null if no config has been saved yet, or the file is unreadable/incomplete. */
export function loadConfig(baseDir: string): BridgeConfig | null {
  const file = configFilePath(baseDir);
  if (!fs.existsSync(file)) return null;
  try {
    const raw = JSON.parse(fs.readFileSync(file, 'utf8')) as Partial<BridgeConfig>;
    if (!raw.algoAkanadeBaseUrl || !raw.algoAkanadeEmail || !raw.algoAkanadePassword) return null;
    return {
      algoAkanadeBaseUrl: String(raw.algoAkanadeBaseUrl).replace(/\/+$/, ''),
      algoAkanadeEmail: String(raw.algoAkanadeEmail),
      algoAkanadePassword: String(raw.algoAkanadePassword),
      tallyBaseUrl: raw.tallyBaseUrl ? String(raw.tallyBaseUrl) : DEFAULT_TALLY_BASE_URL,
      bridgePort: raw.bridgePort ? Number(raw.bridgePort) : DEFAULT_BRIDGE_PORT,
    };
  } catch {
    return null;
  }
}

export function saveConfig(baseDir: string, config: BridgeConfig): void {
  fs.writeFileSync(configFilePath(baseDir), JSON.stringify(config, null, 2), 'utf8');
}

/** Dev/CI convenience only — lets the old environment-variable flow keep working for anyone already using it; a real end user never sets these. */
export function loadConfigFromEnv(): BridgeConfig | null {
  const algoAkanadeBaseUrl = process.env.ALGO_AKANADE_BASE_URL;
  const algoAkanadeEmail = process.env.ALGO_AKANADE_EMAIL;
  const algoAkanadePassword = process.env.ALGO_AKANADE_PASSWORD;
  if (!algoAkanadeBaseUrl || !algoAkanadeEmail || !algoAkanadePassword) return null;
  return {
    algoAkanadeBaseUrl: algoAkanadeBaseUrl.replace(/\/+$/, ''),
    algoAkanadeEmail,
    algoAkanadePassword,
    tallyBaseUrl: process.env.TALLY_BASE_URL ?? DEFAULT_TALLY_BASE_URL,
    bridgePort: process.env.BRIDGE_PORT ? Number(process.env.BRIDGE_PORT) : DEFAULT_BRIDGE_PORT,
  };
}
