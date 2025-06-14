import React, { useState, useRef, useEffect, KeyboardEvent } from 'react';
import { Send, Paperclip } from 'lucide-react';
import { useDocumentStore } from '@/stores/documentStore';
import { useDebounce } from '../../hooks/useDebounce';
import styles from './ChatInputBar.module.css';

interface Props {
  projectId: string;
  onSend: (_text: string, _files?: File[]) => void;
  onTyping?: () => void;
  isDisabled?: boolean;
  placeholder?: string;
  suggestions?: string[];
}

const ChatInputBar: React.FC<Props> = ({
  projectId,
  onSend,
  onTyping,
  isDisabled,
  placeholder = 'Type a message…',
  suggestions = [],
}) => {
  const [text, setText] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const { uploadDocument } = useDocumentStore();

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

  // Handle file attachment
  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setAttachedFile(file);
      // Upload immediately
      try {
        await uploadDocument(projectId, file);
        // Add reference in message
        setText(prev => prev + (prev ? '\n' : '') + `[Attached: ${file.name}]`);
      } catch (error) {
        console.error('Failed to upload file:', error);
      }
      setAttachedFile(null);
    }
  };

  // Send helper
  const send = () => {
    const value = text.trim();
    if (!value) return;
    onSend(value, attachedFile ? [attachedFile] : undefined);
    setText('');
    setAttachedFile(null);
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
      {attachedFile && (
        <div className={styles.attachPreview}>
          <span>{attachedFile.name}</span>
          <button onClick={() => setAttachedFile(null)}>×</button>
        </div>
      )}

      <div className="flex items-end gap-2">
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
          <button
            onClick={() => fileRef.current?.click()}
            disabled={isDisabled}
            title="Attach document"
          >
            <Paperclip size={18} />
          </button>
          <input
            ref={fileRef}
            type="file"
            hidden
            onChange={handleFileSelect}
            accept=".pdf,.docx,.doc,.txt,.md,.csv"
          />
          <button
            onClick={send}
            disabled={isDisabled || !text.trim()}
            title="Send message"
          >
            <Send size={18} />
          </button>
        </div>
      </div>

      {showSuggestions && suggestions.length > 0 && (
        <div className={styles.suggestions}>
          {suggestions.map((suggestion, i) => (
            <button
              key={i}
              onClick={() => {
                setText(suggestion);
                setShowSuggestions(false);
              }}
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default ChatInputBar;
