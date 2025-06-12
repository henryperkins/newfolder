import React, { useState, useEffect } from 'react';
import { X, Clock, User, FileText, RotateCcw } from 'lucide-react';
import { useDocumentStore } from '@/stores/documentStore';
import { DocumentVersion } from '@/types/document';
import { Button, Card } from '@/components/common';
import { formatFileSize, formatRelativeTime } from '@/utils';

interface VersionHistoryProps {
  documentId: string;
  onClose: () => void;
}

export const VersionHistory: React.FC<VersionHistoryProps> = ({
  documentId,
  onClose
}) => {
  const [versions, setVersions] = useState<DocumentVersion[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isReverting, setIsReverting] = useState(false);

  const { documents, fetchVersions, revertToVersion } = useDocumentStore();
  const document = documents.get(documentId);

  useEffect(() => {
    loadVersions();
  }, [documentId]);

  const loadVersions = async () => {
    setIsLoading(true);
    try {
      const versionList = await fetchVersions(documentId);
      setVersions(versionList);
    } catch (error) {
      console.error('Failed to load versions:', error);
    }
    setIsLoading(false);
  };

  const handleRevert = async (versionId: string) => {
    if (!window.confirm('Are you sure you want to revert to this version? This will create a new version.')) {
      return;
    }

    setIsReverting(true);
    try {
      await revertToVersion(documentId, versionId);
      await loadVersions(); // Reload versions
    } catch (error) {
      console.error('Failed to revert version:', error);
    }
    setIsReverting(false);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <Card className="w-full max-w-3xl max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Version History</h2>
            {document && (
              <p className="text-sm text-gray-600 mt-1">{document.name}</p>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Version List */}
        <div className="flex-1 overflow-y-auto p-6">
          {isLoading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
            </div>
          ) : versions.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              No version history available
            </div>
          ) : (
            <div className="space-y-4">
              {versions.map((version, index) => {
                const isCurrent = document?.current_version_id === version.id;

                return (
                  <div
                    key={version.id}
                    className={cn(
                      'p-4 rounded-lg border',
                      isCurrent ? 'border-blue-200 bg-blue-50' : 'border-gray-200'
                    )}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <span className="font-medium text-gray-900">
                            Version {version.version_number}
                          </span>
                          {isCurrent && (
                            <span className="px-2 py-1 text-xs font-medium text-blue-700 bg-blue-100 rounded-full">
                              Current
                            </span>
                          )}
                        </div>

                        <div className="flex items-center gap-4 text-sm text-gray-600">
                          <div className="flex items-center gap-1">
                            <Clock className="w-4 h-4" />
                            {formatRelativeTime(version.created_at)}
                          </div>
                          <div className="flex items-center gap-1">
                            <FileText className="w-4 h-4" />
                            {formatFileSize(version.size_bytes)}
                          </div>
                          {version.page_count && (
                            <span>{version.page_count} pages</span>
                          )}
                        </div>

                        {/* Tags */}
                        {version.tags.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-2">
                            {version.tags.map((tag) => (
                              <span
                                key={tag}
                                className="px-2 py-1 text-xs text-gray-600 bg-gray-100 rounded-full"
                              >
                                {tag}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* Actions */}
                      {!isCurrent && (
                        <Button
                          size="sm"
                          variant="secondary"
                          onClick={() => handleRevert(version.id)}
                          disabled={isReverting}
                          className="flex items-center gap-2"
                        >
                          <RotateCcw className="w-4 h-4" />
                          Revert
                        </Button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};
