/* --------------------------------------------------------------------------
 * SearchResults.tsx
 * --------------------------------------------------------------------------
 * Full-featured search results page with:
 *   • race-condition safety (fetch token pattern)
 *   • defensive defaults (never call .length on undefined)
 *   • zero ESLint / TSC errors
 * ------------------------------------------------------------------------ */

import { useState, useEffect, useCallback, useRef } from 'react';
import { Link, useSearchParams, useNavigate } from 'react-router-dom';

import { searchApi } from '@/utils/api';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';

import {
  Brain,
  FileText,
  MessageSquare,
  ChevronRight,
  ArrowLeft,
  Search,
  Loader2,
} from 'lucide-react';

import type {
  SearchResult,
  SearchDocument,
  SearchChat,
  SearchSemanticMatch,
} from './searchTypes';

/* ------------------------------------------------------------------ */
/* Constants / helpers                                                */
/* ------------------------------------------------------------------ */

const EMPTY_RESULTS: SearchResult = {
  documents: [],
  chats: [],
  semantic_matches: [],
};

/* ------------------------------------------------------------------ */
/* Component                                                          */
/* ------------------------------------------------------------------ */

export default function SearchResults() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const query     = (searchParams.get('q') ?? '').trim();
  const projectId = searchParams.get('project') || undefined;

  const [results, setResults] = useState<SearchResult>(EMPTY_RESULTS);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  /* ----------- race-condition guard (monotonically increasing id) -- */
  const fetchIdRef = useRef(0);

  const performSearch = useCallback(async () => {
    if (!query) {
      setResults(EMPTY_RESULTS);
      return;
    }

    const myFetchId = ++fetchIdRef.current;

    setLoading(true);
    setError(null);

    try {
      // `searchApi.search` only takes (query, projectId)
      const data = await searchApi.search(query, projectId);

      if (myFetchId === fetchIdRef.current) {
        // merge with defaults in case backend omits arrays
        setResults({ ...EMPTY_RESULTS, ...data.results });
      }
    } catch (err) {
      console.error('[SearchResults] search error:', err);
      if (myFetchId === fetchIdRef.current) {
        setError('Failed to perform search. Please try again.');
      }
    } finally {
      if (myFetchId === fetchIdRef.current) setLoading(false);
    }
  }, [query, projectId]);

  useEffect(() => {
    performSearch();
  }, [performSearch]);

  /* ----------------------- Render helpers ------------------------- */

  const renderDocument = (doc: SearchDocument) => (
    <Link
      key={doc.id}
      to={`/projects/${doc.project_id}/documents/${doc.id}`}
      className="block"
    >
      <Card className="p-4 hover:shadow-md transition-shadow">
        <div className="flex items-start space-x-3">
          <FileText className="w-5 h-5 text-blue-600 mt-1 shrink-0" />
          <div className="flex-1 min-w-0">
            <h4 className="font-medium text-gray-900 truncate">{doc.title}</h4>
            <p className="text-sm text-gray-600 mt-1 line-clamp-2">{doc.preview}</p>
            <p className="text-xs text-gray-400 mt-2">
              Updated {new Date(doc.updated_at).toLocaleDateString()}
            </p>
          </div>
          <ChevronRight className="w-5 h-5 text-gray-400 shrink-0" />
        </div>
      </Card>
    </Link>
  );

  const renderChat = (chat: SearchChat) => (
    <Link
      key={chat.id}
      to={`/chat?thread=${chat.thread_id}`}
      className="block"
    >
      <Card className="p-4 hover:shadow-md transition-shadow">
        <div className="flex items-start space-x-3">
          <MessageSquare className="w-5 h-5 text-green-600 mt-1 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm text-gray-600 line-clamp-3">{chat.preview}</p>
            <p className="text-xs text-gray-400 mt-2">
              {new Date(chat.created_at).toLocaleDateString()}
            </p>
          </div>
          <ChevronRight className="w-5 h-5 text-gray-400 shrink-0" />
        </div>
      </Card>
    </Link>
  );

  const renderSemantic = (match: SearchSemanticMatch) => {
    const source =
      typeof match.metadata === 'object' && match.metadata !== null
        ? (match.metadata as Record<string, unknown>).source
        : undefined;

    const sourceText = typeof source === 'string' ? source :
                      typeof source === 'number' ? String(source) :
                      source ? String(source) : undefined;

    return (
      <Card key={match.id} className="p-4">
        <div className="flex items-start space-x-3">
          <Brain className="w-5 h-5 text-purple-600 mt-1 shrink-0" />
          <div className="flex-1">
            <p className="text-sm text-gray-600 line-clamp-3">{match.content}</p>
            <div className="flex items-center mt-2 text-xs text-gray-400 space-x-4">
              <span>Relevance: {(match.score * 100).toFixed(0)}%</span>
              {sourceText && <span>Source: {sourceText}</span>}
            </div>
          </div>
        </div>
      </Card>
    );
  };

  /* ----------------------------- UI ------------------------------ */

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="animate-spin w-8 h-8 text-blue-600" />
      </div>
    );
  }

  const nothingFound = Object.values(results).every(arr => arr.length === 0);

  return (
    <div className="max-w-5xl mx-auto p-6">
      {/* Back button */}
      <Button
        variant="ghost"
        size="sm"
        onClick={() => navigate(-1)}
        className="mb-6"
      >
        <ArrowLeft className="w-4 h-4 mr-2" />
        Back
      </Button>

      {/* Heading */}
      <h1 className="text-2xl font-bold text-gray-900 mb-2">Search Results</h1>
      <p className="text-gray-600 mb-8">
        Showing results for &quot;{query}&quot;
        {projectId && (
          <span className="text-sm"> in current project</span>
        )}
      </p>

      {/* Error banner */}
      {error && (
        <Card className="mb-6 p-4 bg-red-50 border-red-200">
          <p className="text-red-700">{error}</p>
        </Card>
      )}

      {/* Results */}
      {!error && (
        <div className="space-y-8">
          {/* Documents */}
          {results.documents.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <FileText className="w-5 h-5 mr-2" />
                Documents ({results.documents.length})
              </h2>
              <div className="space-y-3">
                {results.documents.map(renderDocument)}
              </div>
            </section>
          )}

          {/* Chat messages */}
          {results.chats.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <MessageSquare className="w-5 h-5 mr-2" />
                Chat Messages ({results.chats.length})
              </h2>
              <div className="space-y-3">
                {results.chats.map(renderChat)}
              </div>
            </section>
          )}

          {/* Semantic matches */}
          {results.semantic_matches.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Brain className="w-5 h-5 mr-2" />
                Related Content ({results.semantic_matches.length})
              </h2>
              <div className="space-y-3">
                {results.semantic_matches.map(renderSemantic)}
              </div>
            </section>
          )}

          {/* Nothing found */}
          {nothingFound && (
            <Card className="text-center py-12">
              <Search className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">No results found for your search.</p>
              <p className="text-sm text-gray-500 mt-2">
                Try different keywords or search in all projects.
              </p>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
