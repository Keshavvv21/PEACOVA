import { type JSX } from 'react';
import { type UseShikiChatOptions } from 'algo-shiki-react-core';
export interface ShikiChatWidgetProps extends UseShikiChatOptions {
    title?: string;
    placeholder?: string;
}
export declare function ShikiChatWidget(props: ShikiChatWidgetProps): JSX.Element;
