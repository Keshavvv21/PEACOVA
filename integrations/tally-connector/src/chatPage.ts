// The actual "popup chat" a non-technical user opens — a single,
// dependency-free HTML page (no build step, no bundler) served
// directly by the bridge server. Plain chat via POST /chat, plus two
// one-click "quick actions" that call the Tally-aware endpoints
// (/analyze-ledgers, /analyze-daybook) this bridge already exposes.
// Voice uses the browser's own Web Speech API directly — the same
// approach as algo-akanade-voice's browser adapter, inlined here so
// this page has zero dependencies and works from a plain file:// or
// http://localhost URL with nothing to install.
export function renderChatPage(): string {
  return `<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Algo AI for Tally</title>
<style>
  html, body { height: 100%; margin: 0; }
  body { font-family: system-ui, sans-serif; background: #f3f4f6; display: flex; justify-content: center; }
  #app { width: 100%; max-width: 480px; height: 100vh; display: flex; flex-direction: column; background: #fff; border-left: 1px solid #e5e7eb; border-right: 1px solid #e5e7eb; }
  #header { padding: 14px 16px; font-weight: 700; font-size: 16px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
  #header a { font-size: 12px; color: #666; text-decoration: none; }
  #messages { flex: 1; overflow-y: auto; padding: 14px 16px; }
  .row { display: flex; margin-bottom: 10px; }
  .row.user { justify-content: flex-end; }
  .bubble { max-width: 82%; padding: 8px 12px; border-radius: 8px; white-space: pre-wrap; font-size: 14px; line-height: 1.4; }
  .row.user .bubble { background: #1a56db; color: #fff; }
  .row.assistant .bubble { background: #f3f4f6; color: #111; }
  .bubble img { max-width: 100%; margin-top: 8px; border-radius: 4px; }
  #quick { display: flex; gap: 8px; padding: 8px 16px; border-top: 1px solid #eee; }
  #quick button { flex: 1; padding: 8px; border: 1px solid #ddd; border-radius: 6px; background: #fff; font-size: 12px; cursor: pointer; }
  #quick button:hover { background: #f3f4f6; }
  #inputRow { display: flex; gap: 8px; padding: 10px 16px; border-top: 1px solid #eee; }
  #input { flex: 1; padding: 9px 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 14px; }
  #inputRow button { padding: 9px 14px; border: none; border-radius: 6px; background: #1a56db; color: #fff; font-weight: 600; cursor: pointer; }
  #mic { background: #f3f4f6 !important; color: #111 !important; }
  #inputRow button:disabled { opacity: 0.5; cursor: default; }
  .error { color: #b91c1c; font-size: 13px; margin-top: 6px; }
</style>
</head>
<body>
<div id="app">
  <div id="header"><span>Algo AI</span><a href="/setup">Settings</a></div>
  <div id="messages"></div>
  <div id="quick">
    <button id="qLedgers" type="button">Analyze my ledgers</button>
    <button id="qDaybook" type="button">Analyze this month's Day Book</button>
  </div>
  <div id="inputRow">
    <input id="input" placeholder="Ask Algo AI..." autocomplete="off" />
    <button id="mic" type="button" title="Speak">MIC</button>
    <button id="send" type="button">Send</button>
  </div>
</div>
<script>
(function () {
  var messagesEl = document.getElementById('messages');
  var inputEl = document.getElementById('input');
  var sendBtn = document.getElementById('send');
  var micBtn = document.getElementById('mic');

  function addMessage(role, text) {
    var row = document.createElement('div');
    row.className = 'row ' + role;
    var bubble = document.createElement('div');
    bubble.className = 'bubble';
    var imgMatch = /!\\[[^\\]]*\\]\\((data:image\\/[a-zA-Z]+;base64,[^)]+)\\)/.exec(text);
    if (imgMatch) {
      var before = text.slice(0, imgMatch.index).trim();
      if (before) bubble.appendChild(document.createTextNode(before));
      var img = document.createElement('img');
      img.src = imgMatch[1];
      bubble.appendChild(document.createElement('br'));
      bubble.appendChild(img);
    } else {
      bubble.textContent = text;
    }
    row.appendChild(bubble);
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return bubble;
  }

  function addError(message) {
    var el = document.createElement('div');
    el.className = 'error';
    el.textContent = message;
    messagesEl.appendChild(el);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function stripForSpeech(text) {
    return text
      .replace(/!\\[[^\\]]*\\]\\([^)]*\\)/g, '')
      .replace(/\`\`\`[\\s\\S]*?\`\`\`/g, ' code omitted ')
      .replace(/\`([^\`]+)\`/g, '$1')
      .replace(/\\s+/g, ' ')
      .trim();
  }

  function speak(text) {
    if (!window.speechSynthesis) return;
    var cleaned = stripForSpeech(text);
    if (!cleaned) return;
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(cleaned));
  }

  var sessionId = 'tally-popup-' + Date.now();

  function setBusy(busy) {
    sendBtn.disabled = busy;
    inputEl.disabled = busy;
  }

  function send(text) {
    if (!text || !text.trim()) return;
    addMessage('user', text);
    setBusy(true);
    fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, sessionId: sessionId }),
    })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        setBusy(false);
        if (data.error) { addError(data.error); return; }
        addMessage('assistant', data.reply);
        speak(data.reply);
      })
      .catch(function (err) {
        setBusy(false);
        addError('Could not reach Algo AI: ' + err);
      });
  }

  sendBtn.addEventListener('click', function () {
    var text = inputEl.value;
    inputEl.value = '';
    send(text);
  });
  inputEl.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') sendBtn.click();
  });

  var SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognitionCtor) {
    micBtn.style.display = 'none';
  } else {
    micBtn.addEventListener('click', function () {
      var recognizer = new SpeechRecognitionCtor();
      recognizer.lang = 'en-US';
      recognizer.onresult = function (event) {
        send(event.results[0][0].transcript);
      };
      recognizer.onerror = function () {};
      recognizer.start();
    });
  }

  function runTallyAction(path, buildBody, label) {
    var companyName = window.prompt('Company name (exactly as it appears in Tally):');
    if (!companyName) return;
    var body = buildBody(companyName);
    addMessage('user', label + ' — ' + companyName);
    setBusy(true);
    fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        setBusy(false);
        if (data.error) { addError(data.error); return; }
        addMessage('assistant', data.reply);
        speak(data.reply);
      })
      .catch(function (err) {
        setBusy(false);
        addError('Could not reach Tally through the bridge: ' + err);
      });
  }

  document.getElementById('qLedgers').addEventListener('click', function () {
    runTallyAction(
      '/analyze-ledgers',
      function (companyName) {
        return { companyName: companyName, question: 'Summarize anything unusual or worth a closer look.' };
      },
      'Analyze ledgers',
    );
  });

  document.getElementById('qDaybook').addEventListener('click', function () {
    var fromDate = window.prompt('From date (YYYYMMDD, e.g. 20240401):');
    if (!fromDate) return;
    var toDate = window.prompt('To date (YYYYMMDD, e.g. 20240430):');
    if (!toDate) return;
    var companyName = window.prompt('Company name (exactly as it appears in Tally):');
    if (!companyName) return;
    addMessage('user', 'Analyze Day Book ' + fromDate + '-' + toDate + ' — ' + companyName);
    setBusy(true);
    fetch('/analyze-daybook', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ companyName: companyName, fromDate: fromDate, toDate: toDate, question: 'Summarize this period.' }),
    })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        setBusy(false);
        if (data.error) { addError(data.error); return; }
        addMessage('assistant', data.reply);
        speak(data.reply);
      })
      .catch(function (err) {
        setBusy(false);
        addError('Could not reach Tally through the bridge: ' + err);
      });
  });

  addMessage('assistant', 'Hi! Ask me anything, or use a quick action below for your Tally ledgers.');
})();
</script>
</body>
</html>`;
}
