import { type JSX } from 'react';
import { type UseMagnetoChatOptions } from 'algo-magneto-react-core';
export interface MagnetoChatWidgetProps extends UseMagnetoChatOptions {
    title?: string;
    placeholder?: string;
}
export declare function MagnetoChatWidget(props: MagnetoChatWidgetProps): JSX.Element;
