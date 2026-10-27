#!/usr/bin/env node
// The single program a non-technical user runs: double-click it (or
// run it from a terminal, same thing), and it opens a browser window —
// first-time setup if nothing is configured yet, the chat window if it
// is. No npm, no environment variables, no editing files by hand.
import path from 'node:path';
import { createBridgeServer } from './server';
import { DEFAULT_BRIDGE_PORT, loadConfig, loadConfigFromEnv, saveConfig } from './config';
import { openInBrowser } from './openBrowser';

// When packaged by pkg into a standalone executable, process.pkg is
// set and the executable's own directory is where the config file
// should live (next to the .exe, easy for a user to find). Otherwise
// (running from source with `npm start`), use the current directory.
const isPackaged = Boolean((process as unknown as { pkg?: unknown }).pkg);
const baseDir = isPackaged ? path.dirname(process.execPath) : process.cwd();

// A dev/CI convenience: if environment variables are set and no config
// file exists yet, seed the file from them once so server.ts has one
// single source of truth (the file) from here on.
const envConfig = loadConfigFromEnv();
if (envConfig && !loadConfig(baseDir)) {
  saveConfig(baseDir, envConfig);
}

const config = loadConfig(baseDir);
const port = config?.bridgePort ?? DEFAULT_BRIDGE_PORT;

const server = createBridgeServer({ baseDir });
server.listen(port, () => {
  const url = `http://localhost:${port}${config ? '/' : '/setup'}`;
  console.log('Algo AI for Tally is running.');
  console.log(config ? `Chat: ${url}` : `First-time setup: ${url}`);
  console.log('Leave this window open — closing it stops Algo AI for Tally.');
  openInBrowser(url);
});
