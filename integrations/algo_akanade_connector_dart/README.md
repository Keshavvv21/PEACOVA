# algo_akanade_connector (Dart)

A plain Dart client for ALGO AKANADE's existing REST API — the Dart twin of
[`../algo-akanade-connector`](../algo-akanade-connector) (TypeScript).
For any Flutter/Dart app **other than** `tool_layer_akanade/mobile`
(ALGO AKANADE's own official mobile app, which already has its own separate
client in `chat_repository.dart`) that wants the same integration
without reimplementing it.

## Install

Not published to pub.dev — use as a local path dependency:

```yaml
dependencies:
  algo_akanade_connector:
    path: ../path/to/integrations/algo_akanade_connector_dart
```

## Usage

```dart
import 'package:algo_akanade_connector/algo_akanade_connector.dart';

final client = AkanadeClient(baseUrl: 'https://your-host/chatbot');
await client.login('you@company.com', 'your-password');

final reply = await client.chat('Backtest AAPL for the last 10 years', 'session-1');
print(reply.reply); // includes CAGR/volatility/Sharpe and an embedded chart image (markdown)

// Feed data back in as a Train Algo AI example:
await client.trainAlgoAiText('trading', 'Q3 summary', someExtractedText);

// Admin token only — returns null for anyone else instead of throwing:
final jobs = await client.runningTrainingJobs();
```

## What this does NOT do

Same scope limits as the TypeScript connector: no offline retry queue,
no structured (non-chat) data endpoints beyond what
`/api/chat/message` already returns as prose, no token refresh (catch
a `AkanadeApiException` with `status == 401` and call `login()` again).

## Tests

```bash
dart pub get
dart test
```

7 tests, all against a mocked `http.Client` (`package:http/testing.dart`'s
`MockClient`) — no live backend required.
