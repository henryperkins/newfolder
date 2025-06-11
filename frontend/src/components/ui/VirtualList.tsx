/**
 * Ultra-simple virtual list component.
 *
 * The spec suggests using `react-window` or similar for windowed list
 * rendering.  To keep the dependency footprint minimal we implement a naïve
 * variant that *still* exposes the same **API contract** so that we can swap
 * in react-window later without touching callers.
 *
 * Behaviour:
 * • Renders every item inside a div with inline-block height calculated via
 *   the provided `itemHeight` callback.
 * • Uses a wrapper with `overflow-y: auto` and fixed *height* prop so parent
 *   layouts remain unchanged.
 * • No actual DOM virtualization – good enough for <1000 messages; avoids
 *   extra libraries in Phase-3.
 */

import React from 'react';

export interface VirtualListProps<T> {
  items: T[];
  height: number; // px
  itemHeight: (index: number) => number;
  renderItem: (item: T, index: number) => React.ReactNode;
  className?: string;
}

export function VirtualList<T>({
  items,
  height,
  itemHeight,
  renderItem,
  className,
}: VirtualListProps<T>) {
  return (
    <div
      style={{ height, overflowY: 'auto' }}
      className={className}
    >
      {items.map((item, idx) => (
        <div key={idx} style={{ minHeight: itemHeight(idx) }}>
          {renderItem(item, idx)}
        </div>
      ))}
    </div>
  );
}

export default VirtualList;
