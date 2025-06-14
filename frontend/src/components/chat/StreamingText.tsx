import React, { useState, useEffect, useRef } from 'react';
import Markdown from '../ui/Markdown';
import styles from './StreamingText.module.css';

interface StreamingTextProps {
  text: string;
  isComplete: boolean;
  speed?: number; // chars per second
  onComplete?: () => void;
}

const StreamingText: React.FC<StreamingTextProps> = ({
  text,
  isComplete,
  speed = 30,
  onComplete,
}) => {
  const [displayed, setDisplayed] = useState('');
  const indexRef = useRef(0);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    if (isComplete) {
      setDisplayed(text);
      indexRef.current = text.length;
      onComplete?.();
      return;
    }

    const interval = 1000 / speed;
    timerRef.current = window.setInterval(() => {
      indexRef.current += 1;
      if (indexRef.current >= text.length) {
        clearInterval(timerRef.current as number);
        setDisplayed(text);
        onComplete?.();
      } else {
        setDisplayed(text.slice(0, indexRef.current));
      }
    }, interval);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [text, isComplete, speed, onComplete]);

  return (
    <div className={styles.streamingText}>
      <Markdown content={displayed} />
      {!isComplete && indexRef.current < text.length && <span className={styles.cursor}>▊</span>}
    </div>
  );
};

export default StreamingText;
