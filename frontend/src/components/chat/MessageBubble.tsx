import React, { useState, useEffect, useRef } from 'react';
import type { ChatMessage } from '../../types/chat';
import StreamingText from './StreamingText';
import Markdown from '../ui/Markdown';
import { formatRelativeTime } from '../../utils/date';
import {
  Copy,
  Edit2,
  Trash2,
  RefreshCw,
  User,
  Bot,
} from 'lucide-react';
import styles from './MessageBubble.module.css';

interface MessageBubbleProps {
  message: ChatMessage;
  isStreaming?: boolean;
  isLast?: boolean;
  showAvatar?: boolean;
  focused?: boolean;
  onEdit?: (id: string) => void;
  onDelete?: (id: string) => void;
  onRegenerate?: () => void;
  onCopy?: () => void;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  isStreaming = false,
  isLast = false,
  showAvatar = true,
  focused = false,
  onEdit,
  onDelete,
  onRegenerate,
  onCopy,
}) => {
  const [showActions, setShowActions] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const actionsRef = useRef<HTMLDivElement>(null);

  // Auto-collapse very long messages once streaming finished
  useEffect(() => {
    if (message.content.length > 1000 && !isStreaming) {
      setIsCollapsed(true);
    }
  }, [message.content.length, isStreaming]);

  // Copy helper
  const handleCopy = () => {
    if (onCopy) {
      onCopy();
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    }
  };

  const contentToRender = isCollapsed
    ? message.content.slice(0, 500) + '…'
    : message.content;

  const bubbleCls = `
    ${styles.messageBubble}
    ${message.is_user ? styles.userMessage : styles.assistantMessage}
    ${focused ? styles.focused : ''}
    ${message.is_deleted ? styles.deleted : ''}
  `;

  return (
    <div
      className={styles.messageRow}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      {showAvatar && (
        <div className={styles.avatar}>{message.is_user ? <User size={20} /> : <Bot size={20} />}</div>
      )}

      <div className={bubbleCls}>
        <div className={styles.header}>
          <span className={styles.sender}>{message.is_user ? 'You' : 'Assistant'}</span>
          <span className={styles.timestamp}>{formatRelativeTime(message.created_at)}</span>
          {message.is_edited && <span className={styles.editedLabel}>(edited)</span>}
        </div>

        <div className={styles.contentArea}>
          {message.is_deleted ? (
            <em className={styles.deletedText}>Message deleted</em>
          ) : isStreaming ? (
            <StreamingText text={contentToRender} isComplete={false} />
          ) : (
            <Markdown content={contentToRender} />
          )}
        </div>

        {message.content.length > 1000 && !message.is_deleted && (
          <button
            className={styles.expandBtn}
            onClick={() => setIsCollapsed(!isCollapsed)}
          >
            {isCollapsed ? 'Show more' : 'Show less'}
          </button>
        )}

        {!message.is_user && message.model_used && (
          <div className={styles.metadata}>
            <span className={styles.model}>{message.model_used}</span>
            <span className={styles.tokens}>{message.token_count} tokens</span>
          </div>
        )}

        {showActions && !message.is_deleted && (
          <div ref={actionsRef} className={styles.actions}>
            <button onClick={handleCopy} title="Copy">
              <Copy size={16} />
            </button>
            {copySuccess && <span className={styles.copyFeedback}>Copied!</span>}

            {message.is_user && onEdit && (
              <button onClick={() => onEdit(message.id)} title="Edit">
                <Edit2 size={16} />
              </button>
            )}

            {!message.is_user && isLast && onRegenerate && (
              <button onClick={onRegenerate} title="Regenerate">
                <RefreshCw size={16} />
              </button>
            )}

            {onDelete && (
              <button onClick={() => onDelete(message.id)} title="Delete">
                <Trash2 size={16} />
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default MessageBubble;
