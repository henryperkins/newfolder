import React from 'react';
import { WebSocketStatus } from '../../types/websocket';
import { WifiOff, RefreshCw } from 'lucide-react';
import styles from './ConnectionStatus.module.css';

interface Props {
  status: WebSocketStatus;
  onReconnect?: () => void;
}

const ConnectionStatus: React.FC<Props> = ({ status, onReconnect }) => {
  if (status === WebSocketStatus.CONNECTED) return null;

  const info = {
    [WebSocketStatus.CONNECTING]: {
      icon: <RefreshCw size={16} className={styles.spin} />,
      text: 'Connecting…',
      cls: styles.connecting,
    },
    [WebSocketStatus.DISCONNECTED]: {
      icon: <WifiOff size={16} />,
      text: 'Disconnected',
      cls: styles.disconnected,
      retry: true,
    },
    [WebSocketStatus.ERROR]: {
      icon: <WifiOff size={16} />,
      text: 'Connection error',
      cls: styles.error,
      retry: true,
    },
  }[status];

  if (!info) return null;

  return (
    <div className={`${styles.banner} ${info.cls}`}>
      {info.icon} <span>{info.text}</span>
      {info.retry && onReconnect && (
        <button onClick={onReconnect} className={styles.retryBtn}>
          Retry
        </button>
      )}
    </div>
  );
};

export default ConnectionStatus;
