import React, { useState, useEffect } from 'react';
import { useSearchParams, Link, useNavigate } from 'react-router-dom';
import { Search, FileText, MessageSquare, Brain, ChevronRight, ArrowLeft } from 'lucide-react';
import { searchApi } from '../utils/api';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button';

interface SearchResult {
  documents: Array<{
    id: string;
    title: string;
    preview: string;
    project_id: string;
    updated_at: string;
    type: string;
  }>;
  chats: Array<{
    id: string;
    thread_id: string;
    preview: string;
    created_at: string;
    type: string;
  }>;
  semantic_matches: Array<{
    id: string;
    content: string;
    score: number;
    metadata: any;
    type: string;
  }>;
}

export default function SearchResults() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const query = searchParams.get('q') || '';
  const projectId = searchParams.get('project') || undefined;

  const [results, setResults] = useState<SearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (query) {
      performSearch();
    }
  }, [query, projectId]);

  const performSearch = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await searchApi.search(query, projectId);
      setResults(data.results);
    } catch (err) {
      setError('Failed to perform search. Please try again.');
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  };

  const renderDocumentResult = (doc: any) => (
    <Link
      key={doc.id}
      to={`/projects/${doc.project_id}/documents/${doc.id}`}
      className="block"
    >
      <Card className="p-4 hover:shadow-md transition-shadow">
        <div className="flex items-start space-x-3">
          <FileText className="w-5 h-5 text-blue-600 mt-1 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <h4 className="font-medium text-gray-900 truncate">{doc.title}</h4>
            <p className="text-sm text-gray-600 mt-1 line-clamp-2">{doc.preview}</p>
            <p className="text-xs text-gray-400 mt-2">
              Updated {new Date(doc.updated_at).toLocaleDateString()}
            </p>
          </div>
          <ChevronRight className="w-5 h-5 text-gray-400 flex-shrink-0" />
        </div>
      </Card>
    </Link>
  );

  const renderChatResult = (chat: any) => (
    <Link
      key={chat.id}
      to={`/chat?thread=${chat.thread_id}`}
      className="block"
    >
      <Card className="p-4 hover:shadow-md transition-shadow">
        <div className="flex items-start space-x-3">
          <MessageSquare className="w-5 h-5 text-green-600 mt-1 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm text-gray-600 line-clamp-3">{chat.preview}</p>
            <p className="text-xs text-gray-400 mt-2">
              {new Date(chat.created_at).toLocaleDateString()}
            </p>
          </div>
          <ChevronRight className="w-5 h-5 text-gray-400 flex-shrink-0" />
        </div>
      </Card>
    </Link>
  );

  const renderSemanticResult = (match: any) => (
    <Card key={match.id} className="p-4">
      <div className="flex items-start space-x-3">
        <Brain className="w-5 h-5 text-purple-600 mt-1 flex-shrink-0" />
        <div className="flex-1">
          <p className="text-sm text-gray-600 line-clamp-3">{match.content}</p>
          <div className="flex items-center mt-2 text-xs text-gray-400">
            <span>Relevance: {(match.score * 100).toFixed(0)}%</span>
            {match.metadata?.source && (
              <span className="ml-4">Source: {match.metadata.source}</span>
            )}
          </div>
        </div>
      </div>
    </Card>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto p-6">
      <div className="mb-8">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigate(-1)}
          className="mb-4"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back
        </Button>

        <h1 className="text-2xl font-bold text-gray-900 mb-2">
          Search Results
        </h1>
        <p className="text-gray-600">
          Showing results for &quot;{query}&quot;
          {projectId && <span className="text-sm"> in current project</span>}
        </p>
      </div>

      {error && (
        <Card className="mb-6 p-4 bg-red-50 border-red-200">
          <p className="text-red-700">{error}</p>
        </Card>
      )}

      {results && (
        <div className="space-y-8">
          {/* Documents Section */}
          {results.documents.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <FileText className="w-5 h-5 mr-2" />
                Documents ({results.documents.length})
              </h2>
              <div className="space-y-3">
                {results.documents.map(renderDocumentResult)}
              </div>
            </section>
          )}

          {/* Chat Messages Section */}
          {results.chats.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <MessageSquare className="w-5 h-5 mr-2" />
                Chat Messages ({results.chats.length})
              </h2>
              <div className="space-y-3">
                {results.chats.map(renderChatResult)}
              </div>
            </section>
          )}

          {/* Semantic Matches Section */}
          {results.semantic_matches.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Brain className="w-5 h-5 mr-2" />
                Related Content ({results.semantic_matches.length})
              </h2>
              <div className="space-y-3">
                {results.semantic_matches.map(renderSemanticResult)}
              </div>
            </section>
          )}

          {/* No Results */}
          {Object.values(results).every(arr => arr.length === 0) && (
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
