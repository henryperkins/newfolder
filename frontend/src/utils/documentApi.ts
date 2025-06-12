import { api } from './api';
import {
  Document,
  DocumentVersion,
  DocumentUploadProgress
} from '@/types/document';

export const documentApi = {
  async getDocuments(
    projectId: string,
    status?: 'processing' | 'indexed' | 'error',
    includeVersions = false
  ): Promise<{ documents: Document[]; total: number }> {
    const params = new URLSearchParams();
    if (status) params.append('status', status);
    if (includeVersions) params.append('include_versions', 'true');

    const response = await api.get(`/projects/${projectId}/documents?${params}`);
    return response.data;
  },

  async getDocument(projectId: string, documentId: string): Promise<Document> {
    const response = await api.get(`/projects/${projectId}/documents/${documentId}`);
    return response.data;
  },

  async uploadDocument(
    projectId: string,
    file: File,
    onProgress?: (percent: number) => void,
    overwriteExisting = false
  ): Promise<{ document_id: string; status: string; message: string }> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('overwrite_existing', overwriteExisting.toString());

    const response = await api.post(`/projects/${projectId}/documents`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress?.(percent);
        }
      },
    });

    return response.data;
  },

  async updateDocument(
    documentId: string,
    updates: { name?: string; tags?: string[] }
  ): Promise<Document> {
    // Get project ID from document
    const doc = await this.getDocumentInfo(documentId);
    const response = await api.patch(
      `/projects/${doc.project_id}/documents/${documentId}`,
      updates
    );
    return response.data;
  },

  async deleteDocument(documentId: string): Promise<void> {
    // Get project ID from document
    const doc = await this.getDocumentInfo(documentId);
    await api.delete(`/projects/${doc.project_id}/documents/${documentId}`);
  },

  async getDocumentVersions(documentId: string): Promise<DocumentVersion[]> {
    // Get project ID from document
    const doc = await this.getDocumentInfo(documentId);
    const response = await api.get(
      `/projects/${doc.project_id}/documents/${documentId}/versions`
    );
    return response.data;
  },

  async revertDocumentVersion(
    documentId: string,
    targetVersionId: string
  ): Promise<{ document_id: string; new_version_id: string; message: string }> {
    // Get project ID from document
    const doc = await this.getDocumentInfo(documentId);
    const response = await api.post(
      `/projects/${doc.project_id}/documents/${documentId}/revert`,
      { target_version_id: targetVersionId }
    );
    return response.data;
  },

  // Helper to get document info (including project ID)
  async getDocumentInfo(documentId: string): Promise<{ project_id: string }> {
    // This would be cached in a real app
    const response = await api.get(`/documents/${documentId}/info`);
    return response.data;
  }
};
