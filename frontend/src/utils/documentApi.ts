import { api } from './api';
import {
  Document,
  DocumentVersion,
  // DocumentUploadProgress // Removed, not used
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
    projectId: string,
    documentId: string,
    updates: { name?: string; tags?: string[] }
  ): Promise<Document> {
    const response = await api.patch(
      `/projects/${projectId}/documents/${documentId}`,
      updates,
    );
    return response.data;
  },

  async deleteDocument(projectId: string, documentId: string): Promise<void> {
    await api.delete(`/projects/${projectId}/documents/${documentId}`);
  },

  async getDocumentVersions(
    projectId: string,
    documentId: string,
  ): Promise<DocumentVersion[]> {
    const response = await api.get(
      `/projects/${projectId}/documents/${documentId}/versions`,
    );
    return response.data;
  },

  async revertDocumentVersion(
    projectId: string,
    documentId: string,
    targetVersionId: string,
  ): Promise<{ document_id: string; new_version_id: string; message: string }> {
    const response = await api.post(
      `/projects/${projectId}/documents/${documentId}/revert`,
      { target_version_id: targetVersionId },
    );
    return response.data;
  },

  async downloadDocument(
    projectId: string,
    documentId: string,
    filename: string
  ): Promise<void> {
    const response = await api.get(`/projects/${projectId}/documents/${documentId}/download`, {
      responseType: 'blob',
    });
    
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  },

  // Deprecated helper kept for backward compatibility – will be removed once
  // all call sites pass the projectId explicitly.
  async getDocumentInfo(documentId: string): Promise<{ project_id: string }> {
    console.warn(
      'documentApi.getDocumentInfo() is deprecated – supply projectId instead',
    );
    const response = await api.get(`/documents/${documentId}`);
    return response.data;
  },
};
