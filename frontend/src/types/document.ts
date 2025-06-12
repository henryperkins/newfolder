export interface DocumentVersion {
  id: string;
  document_id: string;
  version_number: number;
  file_path: string;
  file_hash: string;
  size_bytes: number;
  page_count?: number;
  word_count?: number;
  tags: string[];
  suggested_tags: string[];
  chunk_count: number;
  created_at: string;
}

export interface Document {
  id: string;
  project_id: string;
  name: string;
  mime_type: string;
  size_bytes: number;
  status: 'processing' | 'indexed' | 'error';
  error_message?: string;
  current_version_id?: string;
  created_at: string;
  updated_at: string;
  indexed_at?: string;
  versions?: DocumentVersion[];
}

export interface DocumentUploadProgress {
  documentId: string;
  fileName: string;
  progress: number;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  error?: string;
}

export interface DocumentNotification {
  id: string;
  type: 'info' | 'success' | 'error';
  documentId: string;
  documentName: string;
  message: string;
  timestamp: string;
}

export interface RAGSource {
  document_id: string;
  document_name: string;
  version_id: string;
  relevance_score: number;
  preview: string;
}
