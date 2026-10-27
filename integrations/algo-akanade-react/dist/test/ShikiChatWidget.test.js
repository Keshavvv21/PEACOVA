"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const jsx_runtime_1 = require("react/jsx-runtime");
const strict_1 = __importDefault(require("node:assert/strict"));
const node_test_1 = require("node:test");
const react_test_renderer_1 = require("react-test-renderer");
const ShikiChatWidget_1 = require("../src/ShikiChatWidget");
function fakeFetch(handler) {
    return (async (input, init = {}) => {
        const { status, body } = handler(String(input), init);
        return { ok: status >= 200 && status < 300, status, statusText: String(status), text: async () => JSON.stringify(body) };
    });
}
function findInput(root) {
    return root.findByType('input');
}
function findButtonByText(root, text) {
    return root.findAll((node) => node.type === 'button' && node.children.join('') === text)[0];
}
(0, node_test_1.test)('renders the title and no messages initially', async () => {
    const fetchImpl = fakeFetch(() => ({ status: 200, body: {} }));
    let renderer;
    await (0, react_test_renderer_1.act)(async () => {
        renderer = (0, react_test_renderer_1.create)((0, jsx_runtime_1.jsx)(ShikiChatWidget_1.ShikiChatWidget, { baseUrl: "https://example.test", accessToken: "tok", fetchImpl: fetchImpl }));
    });
    const text = renderer.root.findAllByType('div').map((n) => n.children).flat();
    strict_1.default.ok(text.includes('Algo AI'));
});
(0, node_test_1.test)('typing into the input and clicking Send sends the message and renders both turns', async () => {
    let sentMessage = '';
    const fetchImpl = fakeFetch((url, init) => {
        strict_1.default.ok(url.endsWith('/api/chat/message'));
        sentMessage = JSON.parse(init.body).message;
        return { status: 200, body: { reply: 'Backtest done.', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } };
    });
    let renderer;
    await (0, react_test_renderer_1.act)(async () => {
        renderer = (0, react_test_renderer_1.create)((0, jsx_runtime_1.jsx)(ShikiChatWidget_1.ShikiChatWidget, { baseUrl: "https://example.test", accessToken: "tok", fetchImpl: fetchImpl }));
    });
    const input = findInput(renderer.root);
    (0, react_test_renderer_1.act)(() => {
        input.props.onChange({ target: { value: 'Backtest AAPL' } });
    });
    const sendButton = findButtonByText(renderer.root, 'Send');
    await (0, react_test_renderer_1.act)(async () => {
        await sendButton.props.onClick();
    });
    strict_1.default.equal(sentMessage, 'Backtest AAPL');
    const bubbleTexts = renderer.root.findAllByType('div').map((n) => n.children.join(''));
    strict_1.default.ok(bubbleTexts.some((t) => t.includes('Backtest AAPL')));
    strict_1.default.ok(bubbleTexts.some((t) => t.includes('Backtest done.')));
});
(0, node_test_1.test)('an embedded base64 chart image is rendered as an <img>, not raw markdown text', async () => {
    const fetchImpl = fakeFetch(() => ({
        status: 200,
        body: {
            reply: 'Strategy return 12%.\n\n![AAPL equity curve](data:image/png;base64,iVBORw0KGgo=)',
            pending_confirmation: false,
            confirm_prompt: null,
            provider: 'groq',
        },
    }));
    let renderer;
    await (0, react_test_renderer_1.act)(async () => {
        renderer = (0, react_test_renderer_1.create)((0, jsx_runtime_1.jsx)(ShikiChatWidget_1.ShikiChatWidget, { baseUrl: "https://example.test", accessToken: "tok", fetchImpl: fetchImpl }));
    });
    const input = findInput(renderer.root);
    (0, react_test_renderer_1.act)(() => {
        input.props.onChange({ target: { value: 'Backtest AAPL' } });
    });
    const sendButton = findButtonByText(renderer.root, 'Send');
    await (0, react_test_renderer_1.act)(async () => {
        await sendButton.props.onClick();
    });
    const images = renderer.root.findAllByType('img');
    strict_1.default.equal(images.length, 1);
    strict_1.default.equal(images[0].props.src, 'data:image/png;base64,iVBORw0KGgo=');
});
(0, node_test_1.test)('a pendingConfirmation reply shows Yes/No buttons, and clicking Yes sends "yes"', async () => {
    let turn = 0;
    const sentMessages = [];
    const fetchImpl = fakeFetch((url, init) => {
        turn += 1;
        sentMessages.push(JSON.parse(init.body).message);
        if (turn === 1) {
            return { status: 200, body: { reply: 'Delete it?', pending_confirmation: true, confirm_prompt: 'Delete it?', provider: 'groq' } };
        }
        return { status: 200, body: { reply: 'Deleted.', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } };
    });
    let renderer;
    await (0, react_test_renderer_1.act)(async () => {
        renderer = (0, react_test_renderer_1.create)((0, jsx_runtime_1.jsx)(ShikiChatWidget_1.ShikiChatWidget, { baseUrl: "https://example.test", accessToken: "tok", fetchImpl: fetchImpl }));
    });
    const input = findInput(renderer.root);
    (0, react_test_renderer_1.act)(() => {
        input.props.onChange({ target: { value: 'delete my task' } });
    });
    await (0, react_test_renderer_1.act)(async () => {
        await findButtonByText(renderer.root, 'Send').props.onClick();
    });
    const yesButton = findButtonByText(renderer.root, 'Yes');
    strict_1.default.ok(yesButton);
    await (0, react_test_renderer_1.act)(async () => {
        await yesButton.props.onClick();
    });
    strict_1.default.deepEqual(sentMessages, ['delete my task', 'yes']);
});
(0, node_test_1.test)('the mic button is not rendered when the environment has no Web Speech API (this test runtime)', async () => {
    const fetchImpl = fakeFetch(() => ({ status: 200, body: {} }));
    let renderer;
    await (0, react_test_renderer_1.act)(async () => {
        renderer = (0, react_test_renderer_1.create)((0, jsx_runtime_1.jsx)(ShikiChatWidget_1.ShikiChatWidget, { baseUrl: "https://example.test", accessToken: "tok", fetchImpl: fetchImpl }));
    });
    const micButtons = renderer.root.findAll((node) => node.type === 'button' && node.props.title === 'Speak');
    strict_1.default.equal(micButtons.length, 0);
});
