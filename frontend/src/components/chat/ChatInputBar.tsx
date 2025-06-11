import React, { useState, useRef, useEffect, KeyboardEvent } from 'react';
import { Send, Paperclip } from 'lucide-react';
import { useDebounce } from '../../hooks/useDebounce';
import styles from './ChatInputBar.module.css';

interface Props {
  onSend: (text: string, files?: File[]) => void;
  onTyping?: () => void;
  isDisabled?: boolean;
  placeholder?: string;
  suggestions?: string[];
}

const ChatInputBar: React.FC<Props> = ({
  onSend,
  onTyping,
  isDisabled,
  placeholder = 'Type a message…',
  suggestions = [],
}) => {
  const [text, setText] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [attachments, setAttachments] = useState<File[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Debounce typing notification
  const debounced = useDebounce(text, 400);
  useEffect(() => {
    if (debounced && onTyping) onTyping();
  }, [debounced, onTyping]);

  // Auto-size textarea
  useEffect(() => {
    if (!textareaRef.current) return;
    textareaRef.current.style.height = 'auto';
    textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
  }, [text]);

  // Send helper
  const send = () => {
    const value = text.trim();
    if (!value) return;
    onSend(value, attachments);
    setText('');
    setAttachments([]);
    setShowSuggestions(false);
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  };

  // Key handling
  const handleKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className={styles.wrapper}>
      {attachments.length > 0 && (
        <div className={styles.attachPreview}>
          {attachments.map((f) => (
            <span key={f.name}>{f.name}</span>
          ))}
        </div>
      )}
      <textarea
        ref={textareaRef}
        className={styles.textarea}
        placeholder={placeholder}
        value={text}
        disabled={isDisabled}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKey}
      />
      <div className={styles.actions}>
        <button onClick={() => fileRef.current?.click()} disabled={isDisabled}>
          <Paperclip size={18} />
        </button>
        <input
          ref={fileRef}
          type="file"
          multiple
          hidden
          onChange={(e) => setAttachments(Array.from(e.target.files || []))}
        />
        <button onClick={send} disabled={isDisabled || !text.trim()}>
          <Send size={18} />
        </button>
      </div>
      {showSuggestions && suggestions.length > 0 && (
        <div className={styles.suggestions}>
          {suggestions.map((sug, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => {
                setText(sug);
                setShowSuggestions(false);
              }}
            >
              {sug}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default ChatInputBar;
