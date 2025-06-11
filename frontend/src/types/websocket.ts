// Shared WebSocket message enums / types – must stay in sync with backend.

export enum WebSocketStatus {
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  DISCONNECTED = 'disconnected',
  ERROR = 'error',
}

export enum MessageType {
  // From server → client
  NEW_MESSAGE = 'new_message',
  ASSISTANT_MESSAGE_START = 'assistant_message_start',
  STREAM_CHUNK = 'stream_chunk',
  MESSAGE_UPDATED = 'message_updated',
  MESSAGE_DELETED = 'message_deleted',

  // From client → server
  SEND_MESSAGE = 'send_message',
  EDIT_MESSAGE = 'edit_message',
  DELETE_MESSAGE = 'delete_message',
  REGENERATE = 'regenerate',
  TYPING_INDICATOR = 'typing_indicator',
}

export interface BaseWsMessage {
  type: MessageType | string;
  [key: string]: unknown;
}
