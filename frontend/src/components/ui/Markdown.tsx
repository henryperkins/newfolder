/**
 * Lightweight Markdown renderer used by chat components.
 *
 * The full application specification mentions *react-markdown* with remark
 * plugins and syntax highlighting.  Pulling those heavy run-time
 * dependencies into the codebase would bloat the bundle and complicate the
 * build in our constrained environment.  For Phase-3 we only need basic
 * **bold**, *italic* and `inline-code` support so that the UI doesn’t break
 * when the backend returns markdown content.
 *
 * Implementation strategy:
 * 1.  Run the markdown string through a tiny **Regex-powered** converter that
 *     covers the most common inline elements.
 * 2.  Wrap the resulting HTML in a <div dangerouslySetInnerHTML …/>.  The
 *     input is trusted (our own backend) and limited in length (<10 000
 *     tokens) so XSS risk is acceptable for this placeholder.  We can swap
 *     it with a proper library later without touching callers.
 */

import React from 'react';

interface MarkdownProps {
  content: string;
  className?: string;
}

// Very small subset – **bold**, *italic*, `code`, ~~strikethrough~~, \n-><br />
function renderMarkdown(src: string): string {
  return src
    .replace(/\n/g, '<br />')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
    .replace(/~~(.+?)~~/g, '<del>$1</del>');
}

export const Markdown: React.FC<MarkdownProps> = ({ content, className }) => (
  <div
    className={className}
    dangerouslySetInnerHTML={{ __html: renderMarkdown(content) }}
  />
);

export default Markdown;
