"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.DEFAULT_BRIDGE_PORT = exports.DEFAULT_TALLY_BASE_URL = void 0;
exports.configFilePath = configFilePath;
exports.loadConfig = loadConfig;
exports.saveConfig = saveConfig;
exports.loadConfigFromEnv = loadConfigFromEnv;
// A local config file next to the running program — no environment
// variables, no command line flags, so a non-technical user never
// needs a terminal after the very first setup page. Read/written as
// plain JSON so it's easy to find and delete by hand if needed.
const node_fs_1 = __importDefault(require("node:fs"));
const node_path_1 = __importDefault(require("node:path"));
exports.DEFAULT_TALLY_BASE_URL = 'http://localhost:9000';
exports.DEFAULT_BRIDGE_PORT = 4600;
const CONFIG_FILE_NAME = 'algo-tally-config.json';
function configFilePath(baseDir) {
    return node_path_1.default.join(baseDir, CONFIG_FILE_NAME);
}
/** Returns null if no config has been saved yet, or the file is unreadable/incomplete. */
function loadConfig(baseDir) {
    const file = configFilePath(baseDir);
    if (!node_fs_1.default.existsSync(file))
        return null;
    try {
        const raw = JSON.parse(node_fs_1.default.readFileSync(file, 'utf8'));
        if (!raw.magnetoBaseUrl || !raw.magnetoEmail || !raw.magnetoPassword)
            return null;
        return {
            magnetoBaseUrl: String(raw.magnetoBaseUrl).replace(/\/+$/, ''),
            magnetoEmail: String(raw.magnetoEmail),
            magnetoPassword: String(raw.magnetoPassword),
            tallyBaseUrl: raw.tallyBaseUrl ? String(raw.tallyBaseUrl) : exports.DEFAULT_TALLY_BASE_URL,
            bridgePort: raw.bridgePort ? Number(raw.bridgePort) : exports.DEFAULT_BRIDGE_PORT,
        };
    }
    catch {
        return null;
    }
}
function saveConfig(baseDir, config) {
    node_fs_1.default.writeFileSync(configFilePath(baseDir), JSON.stringify(config, null, 2), 'utf8');
}
/** Dev/CI convenience only — lets the old environment-variable flow keep working for anyone already using it; a real end user never sets these. */
function loadConfigFromEnv() {
    const magnetoBaseUrl = process.env.MAGNETO_BASE_URL;
    const magnetoEmail = process.env.MAGNETO_EMAIL;
    const magnetoPassword = process.env.MAGNETO_PASSWORD;
    if (!magnetoBaseUrl || !magnetoEmail || !magnetoPassword)
        return null;
    return {
        magnetoBaseUrl: magnetoBaseUrl.replace(/\/+$/, ''),
        magnetoEmail,
        magnetoPassword,
        tallyBaseUrl: process.env.TALLY_BASE_URL ?? exports.DEFAULT_TALLY_BASE_URL,
        bridgePort: process.env.BRIDGE_PORT ? Number(process.env.BRIDGE_PORT) : exports.DEFAULT_BRIDGE_PORT,
    };
}
