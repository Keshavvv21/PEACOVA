#!/usr/bin/env node
"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
// The single program a non-technical user runs: double-click it (or
// run it from a terminal, same thing), and it opens a browser window —
// first-time setup if nothing is configured yet, the chat window if it
// is. No npm, no environment variables, no editing files by hand.
const node_path_1 = __importDefault(require("node:path"));
const server_1 = require("./server");
const config_1 = require("./config");
const openBrowser_1 = require("./openBrowser");
// When packaged by pkg into a standalone executable, process.pkg is
// set and the executable's own directory is where the config file
// should live (next to the .exe, easy for a user to find). Otherwise
// (running from source with `npm start`), use the current directory.
const isPackaged = Boolean(process.pkg);
const baseDir = isPackaged ? node_path_1.default.dirname(process.execPath) : process.cwd();
// A dev/CI convenience: if environment variables are set and no config
// file exists yet, seed the file from them once so server.ts has one
// single source of truth (the file) from here on.
const envConfig = (0, config_1.loadConfigFromEnv)();
if (envConfig && !(0, config_1.loadConfig)(baseDir)) {
    (0, config_1.saveConfig)(baseDir, envConfig);
}
const config = (0, config_1.loadConfig)(baseDir);
const port = config?.bridgePort ?? config_1.DEFAULT_BRIDGE_PORT;
const server = (0, server_1.createBridgeServer)({ baseDir });
server.listen(port, () => {
    const url = `http://localhost:${port}${config ? '/' : '/setup'}`;
    console.log('Algo AI for Tally is running.');
    console.log(config ? `Chat: ${url}` : `First-time setup: ${url}`);
    console.log('Leave this window open — closing it stops Algo AI for Tally.');
    (0, openBrowser_1.openInBrowser)(url);
});
