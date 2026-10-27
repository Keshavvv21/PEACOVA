import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { configFilePath, loadConfig, saveConfig, type BridgeConfig } from '../src/config';

function tempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'algo-tally-config-test-'));
}

test('loadConfig returns null when no config file exists yet', () => {
  assert.equal(loadConfig(tempDir()), null);
});

test('saveConfig then loadConfig round-trips every field', () => {
  const dir = tempDir();
  const config: BridgeConfig = {
    algoAkanadeBaseUrl: 'https://example.test/chatbot',
    algoAkanadeEmail: 'a@b.com',
    algoAkanadePassword: 'secret',
    tallyBaseUrl: 'http://localhost:9000',
    bridgePort: 4600,
  };
  saveConfig(dir, config);
  assert.deepEqual(loadConfig(dir), config);
  assert.ok(fs.existsSync(configFilePath(dir)));
});

test('a trailing slash on algoAkanadeBaseUrl is stripped on load', () => {
  const dir = tempDir();
  saveConfig(dir, {
    algoAkanadeBaseUrl: 'https://example.test/chatbot/',
    algoAkanadeEmail: 'a@b.com',
    algoAkanadePassword: 'secret',
    tallyBaseUrl: 'http://localhost:9000',
    bridgePort: 4600,
  });
  assert.equal(loadConfig(dir)!.algoAkanadeBaseUrl, 'https://example.test/chatbot');
});

test('loadConfig returns null for a file missing a required field', () => {
  const dir = tempDir();
  fs.writeFileSync(configFilePath(dir), JSON.stringify({ algoAkanadeBaseUrl: 'https://example.test' }));
  assert.equal(loadConfig(dir), null);
});

test('loadConfig returns null for invalid JSON instead of throwing', () => {
  const dir = tempDir();
  fs.writeFileSync(configFilePath(dir), 'not json{{{');
  assert.equal(loadConfig(dir), null);
});
