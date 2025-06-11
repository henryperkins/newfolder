import { useEffect, useState } from 'react';

/**
 * Tiny debounce hook – returns the *value* after `delay` ms of no change.
 * Identical signature to the popular `use-debounce` implementation so that
 * we can migrate later without touching callers.
 */
export function useDebounce<T>(value: T, delay = 300): T {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debounced;
}

export default useDebounce;
