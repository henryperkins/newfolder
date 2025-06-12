import React from 'react';
import {
  FileText,
  FilePdf,
  FileSpreadsheet,
  MoreVertical,
  Clock,
  CheckCircle,
  AlertCircle,
  Tag,
  History,
  Trash2,
  Download,
  Edit3
} from 'lucide-react';
import { Document } from '@/types/document';
import { useDocumentStore } from '@/stores/documentStore';
import { Card } from '@/components/common';
import { cn, formatFileSize, formatRelativeTime } from '@/utils';

interface DocumentItemProps {
  document: Document;
  onVersionHistoryClick: () => void;
}

export const DocumentItem: React.FC<DocumentItemProps> = ({
  document,
  onVersionHistoryClick
}) => {
  const {
    applySuggestedTag,
    deleteDocument,
    removeTag
  } = useDocumentStore();

  const [showMenu, setShowMenu] = React.useState(false);
  const [isDeleting, setIsDeleting] = React.useState(false);

  const currentVersion = document.versions?.find(v => v.id === document.current_version_id);

  const getFileIcon = () => {
    if (document.mime_type.includes('pdf')) return FilePdf;
    if (document.mime_type.includes('spreadsheet') || document.mime_type.includes('csv')) return FileSpreadsheet;
    return FileText;
  };

  const getStatusIcon = () => {
    switch (document.status) {
      case 'processing':
        return <Clock className="w-4 h-4 text-yellow-600 animate-pulse" />;
      case 'indexed':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-600" />;
    }
  };

  const getStatusText = () => {
    switch (document.status) {
      case 'processing':
        return 'Processing...';
      case 'indexed':
        return 'Indexed';
      case 'error':
        return 'Error';
    }
  };

  const handleDelete = async () => {
    if (window.confirm(`Are you sure you want to delete "${document.name}"?`)) {
      setIsDeleting(true);
      try {
        await deleteDocument(document.id);
      } catch (error) {
        console.error('Failed to delete document:', error);
      }
      setIsDeleting(false);
    }
  };

  const FileIcon = getFileIcon();

  return (
    <Card className="p-4">
      <div className="flex items-start gap-4">
        {/* File Icon */}
        <div className="flex-shrink-0">
          <FileIcon className="w-10 h-10 text-gray-400" />
        </div>

        {/* Document Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between">
            <div>
              <h3 className="font-medium text-gray-900 truncate">
                {document.name}
              </h3>
              <div className="flex items-center gap-4 mt-1 text-sm text-gray-500">
                <span>{formatFileSize(document.size_bytes)}</span>
                <span>•</span>
                <span>Uploaded {formatRelativeTime(document.created_at)}</span>
                {currentVersion && (
                  <>
                    <span>•</span>
                    <span>Version {currentVersion.version_number}</span>
                  </>
                )}
              </div>
            </div>

            {/* Status & Menu */}
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                {getStatusIcon()}
                <span className={cn(
                  'text-sm font-medium',
                  document.status === 'processing' && 'text-yellow-600',
                  document.status === 'indexed' && 'text-green-600',
                  document.status === 'error' && 'text-red-600'
                )}>
                  {getStatusText()}
                </span>
              </div>

              <div className="relative">
                <button
                  onClick={() => setShowMenu(!showMenu)}
                  className="p-1 hover:bg-gray-100 rounded"
                >
                  <MoreVertical className="w-4 h-4 text-gray-500" />
                </button>

                {showMenu && (
                  <div className="absolute right-0 top-full mt-1 w-48 bg-white rounded-lg shadow-lg border py-1 z-10">
                    <button
                      onClick={onVersionHistoryClick}
                      className="w-full flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                    >
                      <History className="w-4 h-4" />
                      Version History
                    </button>
                    <button
                      onClick={() => {/* TODO: Implement download */}}
                      className="w-full flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                    >
                      <Download className="w-4 h-4" />
                      Download
                    </button>
                    <div className="border-t my-1" />
                    <button
                      onClick={handleDelete}
                      disabled={isDeleting}
                      className="w-full flex items-center gap-2 px-4 py-2 text-sm text-red-600 hover:bg-red-50"
                    >
                      <Trash2 className="w-4 h-4" />
                      {isDeleting ? 'Deleting...' : 'Delete'}
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Error Message */}
          {document.status === 'error' && document.error_message && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-md">
              <p className="text-sm text-red-700">{document.error_message}</p>
            </div>
          )}

          {/* Tags */}
          {currentVersion && (
            <div className="mt-4 flex flex-wrap items-center gap-2">
              {/* Existing Tags */}
              {currentVersion.tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium text-gray-700 bg-gray-100 rounded-full"
                >
                  <Tag className="w-3 h-3" />
                  {tag}
                  <button
                    onClick={() => removeTag(document.id, tag)}
                    className="ml-1 hover:text-gray-900"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))}

              {/* Suggested Tags */}
              {document.status === 'indexed' && currentVersion.suggested_tags.map((tag) => (
                <button
                  key={tag}
                  onClick={() => applySuggestedTag(document.id, tag)}
                  className="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium text-blue-700 bg-blue-50 border border-blue-200 rounded-full hover:bg-blue-100 transition-colors"
                >
                  <Plus className="w-3 h-3" />
                  {tag}
                </button>
              ))}
            </div>
          )}

          {/* Document Stats */}
          {document.status === 'indexed' && currentVersion && (
            <div className="mt-3 flex items-center gap-4 text-xs text-gray-500">
              {currentVersion.page_count && (
                <span>{currentVersion.page_count} pages</span>
              )}
              {currentVersion.word_count && (
                <span>{currentVersion.word_count.toLocaleString()} words</span>
              )}
              {currentVersion.chunk_count > 0 && (
                <span>{currentVersion.chunk_count} indexed chunks</span>
              )}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
};
