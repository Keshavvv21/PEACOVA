// Opens the user's default browser to a local URL — how a
// non-technical user reaches both the first-time setup page and the
// chat window, without ever needing to know a URL or open a terminal.
import { exec } from 'node:child_process';

export function openInBrowser(url: string): void {
  const platform = process.platform;
  const command = platform === 'win32' ? `start "" "${url}"` : platform === 'darwin' ? `open "${url}"` : `xdg-open "${url}"`;
  exec(command, () => {
    // Best-effort — if this fails (headless server, no default browser configured), the console URL printed by cli.ts is still there to open by hand.
  });
}
