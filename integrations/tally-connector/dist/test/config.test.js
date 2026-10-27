"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const strict_1 = __importDefault(require("node:assert/strict"));
const node_fs_1 = __importDefault(require("node:fs"));
const node_os_1 = __importDefault(require("node:os"));
const node_path_1 = __importDefault(require("node:path"));
const node_test_1 = require("node:test");
const config_1 = require("../src/config");
function tempDir() {
    return node_fs_1.default.mkdtempSync(node_path_1.default.join(node_os_1.default.tmpdir(), 'algo-tally-config-test-'));
}
(0, node_test_1.test)('loadConfig returns null when no config file exists yet', () => {
    strict_1.default.equal((0, config_1.loadConfig)(tempDir()), null);
});
(0, node_test_1.test)('saveConfig then loadConfig round-trips every field', () => {
    const dir = tempDir();
    const config = {
        magnetoBaseUrl: 'https://example.test/chatbot',
        magnetoEmail: 'a@b.com',
        magnetoPassword: 'secret',
        tallyBaseUrl: 'http://localhost:9000',
        bridgePort: 4600,
    };
    (0, config_1.saveConfig)(dir, config);
    strict_1.default.deepEqual((0, config_1.loadConfig)(dir), config);
    strict_1.default.ok(node_fs_1.default.existsSync((0, config_1.configFilePath)(dir)));
});
(0, node_test_1.test)('a trailing slash on magnetoBaseUrl is stripped on load', () => {
    const dir = tempDir();
    (0, config_1.saveConfig)(dir, {
        magnetoBaseUrl: 'https://example.test/chatbot/',
        magnetoEmail: 'a@b.com',
        magnetoPassword: 'secret',
        tallyBaseUrl: 'http://localhost:9000',
        bridgePort: 4600,
    });
    strict_1.default.equal((0, config_1.loadConfig)(dir).magnetoBaseUrl, 'https://example.test/chatbot');
});
(0, node_test_1.test)('loadConfig returns null for a file missing a required field', () => {
    const dir = tempDir();
    node_fs_1.default.writeFileSync((0, config_1.configFilePath)(dir), JSON.stringify({ magnetoBaseUrl: 'https://example.test' }));
    strict_1.default.equal((0, config_1.loadConfig)(dir), null);
});
(0, node_test_1.test)('loadConfig returns null for invalid JSON instead of throwing', () => {
    const dir = tempDir();
    node_fs_1.default.writeFileSync((0, config_1.configFilePath)(dir), 'not json{{{');
    strict_1.default.equal((0, config_1.loadConfig)(dir), null);
});
