"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.openInBrowser = openInBrowser;
// Opens the user's default browser to a local URL — how a
// non-technical user reaches both the first-time setup page and the
// chat window, without ever needing to know a URL or open a terminal.
const node_child_process_1 = require("node:child_process");
function openInBrowser(url) {
    const platform = process.platform;
    const command = platform === 'win32' ? `start "" "${url}"` : platform === 'darwin' ? `open "${url}"` : `xdg-open "${url}"`;
    (0, node_child_process_1.exec)(command, () => {
        // Best-effort — if this fails (headless server, no default browser configured), the console URL printed by cli.ts is still there to open by hand.
    });
}
