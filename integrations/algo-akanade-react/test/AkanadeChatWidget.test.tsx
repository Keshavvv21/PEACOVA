import assert from 'node:assert/strict';
import { test } from 'node:test';
import React from 'react';
import { act, create, type ReactTestInstance } from 'react-test-renderer';
import { AkanadeChatWidget } from '../src/AkanadeChatWidget';

function fakeFetch(handler: (url: string, init: RequestInit) => { status: number; body: unknown }): typeof fetch {
  return (async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const { status, body } = handler(String(input), init);
    return { ok: status >= 200 && status < 300, status, statusText: String(status), text: async () => JSON.stringify(body) } as Response;
  }) as typeof fetch;
}

function findInput(root: ReactTestInstance): ReactTestInstance {
  return root.findByType('input');
}
function findButtonByText(root: ReactTestInstance, text: string): ReactTestInstance {
  return root.findAll((node) => node.type === 'button' && node.children.join('') === text)[0];
}

test('renders the title and no messages initially', async () => {
  const fetchImpl = fakeFetch(() => ({ status: 200, body: {} }));
  let renderer: ReturnType<typeof create>;
  await act(async () => {
    renderer = create(<AkanadeChatWidget baseUrl="https://example.test" accessToken="tok" fetchImpl={fetchImpl} />);
  });
  const text = renderer!.root.findAllByType('div').map((n) => n.children).flat();
  assert.ok(text.includes('Algo AI'));
});

test('typing into the input and clicking Send sends the message and renders both turns', async () => {
  let sentMessage = '';
  const fetchImpl = fakeFetch((url, init) => {
    assert.ok(url.endsWith('/api/chat/message'));
    sentMessage = JSON.parse(init.body as string).message;
    return { status: 200, body: { reply: 'Backtest done.', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } };
  });

  let renderer: ReturnType<typeof create>;
  await act(async () => {
    renderer = create(<AkanadeChatWidget baseUrl="https://example.test" accessToken="tok" fetchImpl={fetchImpl} />);
  });

  const input = findInput(renderer!.root);
  act(() => {
    input.props.onChange({ target: { value: 'Backtest AAPL' } });
  });

  const sendButton = findButtonByText(renderer!.root, 'Send');
  await act(async () => {
    await sendButton.props.onClick();
  });

  assert.equal(sentMessage, 'Backtest AAPL');
  const bubbleTexts = renderer!.root.findAllByType('div').map((n) => n.children.join(''));
  assert.ok(bubbleTexts.some((t) => t.includes('Backtest AAPL')));
  assert.ok(bubbleTexts.some((t) => t.includes('Backtest done.')));
});

test('an embedded base64 chart image is rendered as an <img>, not raw markdown text', async () => {
  const fetchImpl = fakeFetch(() => ({
    status: 200,
    body: {
      reply: 'Strategy return 12%.\n\n![AAPL equity curve](data:image/png;base64,iVBORw0KGgo=)',
      pending_confirmation: false,
      confirm_prompt: null,
      provider: 'groq',
    },
  }));
  let renderer: ReturnType<typeof create>;
  await act(async () => {
    renderer = create(<AkanadeChatWidget baseUrl="https://example.test" accessToken="tok" fetchImpl={fetchImpl} />);
  });
  const input = findInput(renderer!.root);
  act(() => {
    input.props.onChange({ target: { value: 'Backtest AAPL' } });
  });
  const sendButton = findButtonByText(renderer!.root, 'Send');
  await act(async () => {
    await sendButton.props.onClick();
  });

  const images = renderer!.root.findAllByType('img');
  assert.equal(images.length, 1);
  assert.equal(images[0].props.src, 'data:image/png;base64,iVBORw0KGgo=');
});

test('a pendingConfirmation reply shows Yes/No buttons, and clicking Yes sends "yes"', async () => {
  let turn = 0;
  const sentMessages: string[] = [];
  const fetchImpl = fakeFetch((url, init) => {
    turn += 1;
    sentMessages.push(JSON.parse(init.body as string).message);
    if (turn === 1) {
      return { status: 200, body: { reply: 'Delete it?', pending_confirmation: true, confirm_prompt: 'Delete it?', provider: 'groq' } };
    }
    return { status: 200, body: { reply: 'Deleted.', pending_confirmation: false, confirm_prompt: null, provider: 'groq' } };
  });
  let renderer: ReturnType<typeof create>;
  await act(async () => {
    renderer = create(<AkanadeChatWidget baseUrl="https://example.test" accessToken="tok" fetchImpl={fetchImpl} />);
  });
  const input = findInput(renderer!.root);
  act(() => {
    input.props.onChange({ target: { value: 'delete my task' } });
  });
  await act(async () => {
    await findButtonByText(renderer!.root, 'Send').props.onClick();
  });

  const yesButton = findButtonByText(renderer!.root, 'Yes');
  assert.ok(yesButton);
  await act(async () => {
    await yesButton.props.onClick();
  });

  assert.deepEqual(sentMessages, ['delete my task', 'yes']);
});

test('the mic button is not rendered when the environment has no Web Speech API (this test runtime)', async () => {
  const fetchImpl = fakeFetch(() => ({ status: 200, body: {} }));
  let renderer: ReturnType<typeof create>;
  await act(async () => {
    renderer = create(<AkanadeChatWidget baseUrl="https://example.test" accessToken="tok" fetchImpl={fetchImpl} />);
  });
  const micButtons = renderer!.root.findAll((node) => node.type === 'button' && node.props.title === 'Speak');
  assert.equal(micButtons.length, 0);
});
