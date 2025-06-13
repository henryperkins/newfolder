import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Upload, FileText, AlertCircle } from 'lucide-react';
import { Document } from '@/types/document';
import { useDocumentStore } from '@/stores/documentStore';
import { Card, Button } from '@/components/common';
import { DocumentUploader } from './DocumentUploader';
import { DocumentItem } from './DocumentItem';
import { VersionHistory } from './VersionHistory';
import { cn } from '@/utils';

export const DocumentManager: React.FC = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const [showUploader, setShowUploader] = useState(false);

  const {
    documents,
    documentsByProject,
    isLoading,
    error,
    selectedDocumentId,
    showVersionHistory,
    fetchDocuments,
    setSelectedDocument,
    setShowVersionHistory
  } = useDocumentStore();

  useEffect(() => {
    if (projectId) {
      fetchDocuments(projectId);
    }
  }, [projectId, fetchDocuments]);

  const projectDocumentIds = documentsByProject.get(projectId || '') || [];
  const projectDocuments = projectDocumentIds
    .map(id => documents.get(id))
    .filter(Boolean) as Document[];

  const handleVersionHistoryOpen = (documentId: string) => {
    setSelectedDocument(documentId);
    setShowVersionHistory(true);
  };

  const handleVersionHistoryClose = () => {
    setSelectedDocument(null);
    setShowVersionHistory(false);
  };

  if (isLoading && projectDocuments.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading documents...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Documents</h2>
          <p className="text-gray-600">
            Upload and manage documents for AI-powered search and context
          </p>
        </div>
        <Button
          onClick={() => setShowUploader(!showUploader)}
          className="flex items-center gap-2"
        >
          <Upload className="w-4 h-4" />
          Upload Documents
        </Button>
      </div>

      {/* Error Message */}
      {error && (
        <Card className="border-red-200 bg-red-50">
          <div className="flex items-center gap-3 text-red-800">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <p>{error}</p>
          </div>
        </Card>
      )}

      {/* Upload Area */}
      {showUploader && (
        <Card>
          <DocumentUploader
            projectId={projectId!}
            onClose={() => setShowUploader(false)}
          />
        </Card>
      )}

      {/* Document List */}
      {projectDocuments.length === 0 ? (
        <Card className="text-center py-12">
          <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No documents yet
          </h3>
          <p className="text-gray-600 mb-6">
            Upload documents to enhance AI responses with your project knowledge
          </p>
          <Button onClick={() => setShowUploader(true)}>
            Upload Your First Document
          </Button>
        </Card>
      ) : (
        <div className="space-y-4">
          {projectDocuments.map((doc) => (
            <DocumentItem
              key={doc.id}
              document={doc}
              onVersionHistoryClick={() => handleVersionHistoryOpen(doc.id)}
            />
          ))}
        </div>
      )}

      {/* Version History Modal */}
      {showVersionHistory && selectedDocumentId && (
        <VersionHistory
          documentId={selectedDocumentId}
          onClose={handleVersionHistoryClose}
        />
      )}
    </div>
  );
};
