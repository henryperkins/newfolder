import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  Document,
  DocumentVersion,
  DocumentUploadProgress,
  DocumentNotification
} from '@/types/document';
import { documentApi } from '@/utils/documentApi';

interface DocumentState {
  // Documents
  documents: Map<string, Document>;
  documentsByProject: Map<string, string[]>; // projectId -> documentIds

  // Upload progress
  uploadProgress: Map<string, DocumentUploadProgress>;

  // Notifications
  notifications: DocumentNotification[];
  unreadNotificationCount: number;

  // UI State
  isLoading: boolean;
  error: string | null;
  selectedDocumentId: string | null;
  showVersionHistory: boolean;
}

interface DocumentActions {
  // Document operations
  fetchDocuments: (projectId: string) => Promise<void>;
  uploadDocument: (
    projectId: string,
    file: File,
    onProgress?: (percent: number) => void,
  ) => Promise<{ document_id: string; status: string; message: string }>;
  updateDocument: (documentId: string, updates: { name?: string; tags?: string[] }) => Promise<void>;
  deleteDocument: (documentId: string) => Promise<void>;

  // Version operations
  fetchVersions: (documentId: string) => Promise<DocumentVersion[]>;
  revertToVersion: (documentId: string, versionId: string) => Promise<void>;

  // Notification operations
  addNotification: (notification: Omit<DocumentNotification, 'id' | 'timestamp'>) => void;
  markNotificationsRead: () => void;
  clearNotifications: () => void;

  // Tag operations
  addTag: (documentId: string, tag: string) => Promise<void>;
  removeTag: (documentId: string, tag: string) => Promise<void>;
  applySuggestedTag: (documentId: string, tag: string) => Promise<void>;

  // UI operations
  setSelectedDocument: (documentId: string | null) => void;
  setShowVersionHistory: (show: boolean) => void;
  updateUploadProgress: (documentId: string, progress: Partial<DocumentUploadProgress>) => void;
  removeUploadProgress: (documentId: string) => void;

  // WebSocket updates
  handleDocumentStatusUpdate: (documentId: string, status: Document['status'], error?: string) => void;
}

type DocumentStore = DocumentState & DocumentActions;

export const useDocumentStore = create<DocumentStore>()(
  persist(
    (set, get) => ({
      // Initial state
      documents: new Map(),
      documentsByProject: new Map(),
      uploadProgress: new Map(),
      notifications: [],
      unreadNotificationCount: 0,
      isLoading: false,
      error: null,
      selectedDocumentId: null,
      showVersionHistory: false,

      // Document operations
      fetchDocuments: async (projectId: string) => {
        set({ isLoading: true, error: null });
        try {
          const response = await documentApi.getDocuments(projectId);
          const docs = new Map<string, Document>();
          const docIds: string[] = [];

          response.documents.forEach((doc: Document) => {
            docs.set(doc.id, doc);
            docIds.push(doc.id);
          });

          set(state => ({
            documents: new Map([...state.documents, ...docs]),
            documentsByProject: new Map([...state.documentsByProject, [projectId, docIds]]),
            isLoading: false
          }));
        } catch (error) {
          set({ error: error instanceof Error ? error.message : 'Failed to fetch documents', isLoading: false });
        }
      },

      uploadDocument: async (projectId: string, file: File, onProgress) => {
        const tempId = `temp-${Date.now()}`;

        // Add to upload progress
        set(state => ({
          uploadProgress: new Map([...state.uploadProgress, [tempId, {
            documentId: tempId,
            fileName: file.name,
            progress: 0,
            status: 'uploading'
          }]])
        }));

        try {
          const uploadResult = await documentApi.uploadDocument(projectId, file, (percent: number) => {
            get().updateUploadProgress(tempId, { progress: percent });
            onProgress?.(percent);
          });

          // Update progress to processing
          get().updateUploadProgress(tempId, {
            documentId: uploadResult.document_id,
            status: 'processing',
            progress: 100
          });

          // Add notification
          get().addNotification({
            type: 'info',
            documentId: uploadResult.document_id,
            documentName: file.name,
            message: 'Document is being processed and indexed...'
          });

          // Fetch updated document list
          await get().fetchDocuments(projectId);

          return uploadResult;
        } catch (error) {
          get().updateUploadProgress(tempId, {
            status: 'error',
            error: error instanceof Error ? error.message : 'Upload failed'
          });
          throw error;
        }
      },

      updateDocument: async (documentId: string, updates) => {
        const stateSnapshot = get();
        const doc = stateSnapshot.documents.get(documentId);
        if (!doc) {
          return;
        }

        try {
          await documentApi.updateDocument(doc.project_id, documentId, updates);
          set((state) => {
            const current = state.documents.get(documentId);
            if (current) {
              state.documents.set(documentId, { ...current, ...updates });
            }
            return { documents: new Map(state.documents) };
          });
        } catch (error) {
          set({
            error:
              error instanceof Error ? error.message : 'Failed to update document',
          });
          throw error;
        }
      },

      deleteDocument: async (documentId: string) => {
        const doc = get().documents.get(documentId);
        if (!doc) return;

        try {
          await documentApi.deleteDocument(doc.project_id, documentId);
          set(state => {
            state.documents.delete(documentId);
            // Remove from project mapping
            state.documentsByProject.forEach((docIds, projectId) => {
              const filtered = docIds.filter(id => id !== documentId);
              state.documentsByProject.set(projectId, filtered);
            });
            return {
              documents: new Map(state.documents),
              documentsByProject: new Map(state.documentsByProject)
            };
          });
        } catch (error) {
          set({ error: error instanceof Error ? error.message : 'Failed to delete document' });
          throw error;
        }
      },

      // Version operations
      fetchVersions: async (documentId: string) => {
        const doc = get().documents.get(documentId);
        if (!doc) throw new Error('Document not found');

        try {
          return await documentApi.getDocumentVersions(doc.project_id, documentId);
        } catch (error) {
          set({
            error:
              error instanceof Error ? error.message : 'Failed to fetch versions',
          });
          throw error;
        }
      },

      revertToVersion: async (documentId: string, versionId: string) => {
        const doc = get().documents.get(documentId);
        if (!doc) throw new Error('Document not found');

        try {
          await documentApi.revertDocumentVersion(doc.project_id, documentId, versionId);
          // Refresh document list for that project
          await get().fetchDocuments(doc.project_id);
        } catch (error) {
          set({
            error:
              error instanceof Error ? error.message : 'Failed to revert version',
          });
          throw error;
        }
      },

      // Notification operations
      addNotification: (notification) => {
        const newNotification: DocumentNotification = {
          ...notification,
          id: `notif-${Date.now()}`,
          timestamp: new Date().toISOString()
        };

        set(state => ({
          notifications: [newNotification, ...state.notifications].slice(0, 50), // Keep last 50
          unreadNotificationCount: state.unreadNotificationCount + 1
        }));
      },

      markNotificationsRead: () => {
        set({ unreadNotificationCount: 0 });
      },

      clearNotifications: () => {
        set({ notifications: [], unreadNotificationCount: 0 });
      },

      // Tag operations
      addTag: async (documentId: string, tag: string) => {
        const doc = get().documents.get(documentId);
        if (!doc || !doc.current_version_id) return;

        const currentVersion = doc.versions?.find(v => v.id === doc.current_version_id);
        if (!currentVersion) return;

        const newTags = [...currentVersion.tags, tag];
        await get().updateDocument(documentId, { tags: newTags });
      },

      removeTag: async (documentId: string, tag: string) => {
        const doc = get().documents.get(documentId);
        if (!doc || !doc.current_version_id) return;

        const currentVersion = doc.versions?.find(v => v.id === doc.current_version_id);
        if (!currentVersion) return;

        const newTags = currentVersion.tags.filter(t => t !== tag);
        await get().updateDocument(documentId, { tags: newTags });
      },

      applySuggestedTag: async (documentId: string, tag: string) => {
        await get().addTag(documentId, tag);

        // Remove from suggested tags in UI
        set(state => {
          const doc = state.documents.get(documentId);
          if (doc && doc.current_version_id && doc.versions) {
            const versionIndex = doc.versions.findIndex(v => v.id === doc.current_version_id);
            if (versionIndex !== -1) {
              doc.versions[versionIndex].suggested_tags = doc.versions[versionIndex].suggested_tags.filter(t => t !== tag);
              state.documents.set(documentId, { ...doc });
            }
          }
          return { documents: new Map(state.documents) };
        });
      },

      // UI operations
      setSelectedDocument: (documentId) => {
        set({ selectedDocumentId: documentId });
      },

      setShowVersionHistory: (show) => {
        set({ showVersionHistory: show });
      },

      updateUploadProgress: (documentId, progress) => {
        set(state => {
          const current = state.uploadProgress.get(documentId);
          if (current) {
            state.uploadProgress.set(documentId, { ...current, ...progress });
          }
          return { uploadProgress: new Map(state.uploadProgress) };
        });
      },

      removeUploadProgress: (documentId) => {
        set(state => {
          state.uploadProgress.delete(documentId);
          return { uploadProgress: new Map(state.uploadProgress) };
        });
      },

      // WebSocket updates
      handleDocumentStatusUpdate: (documentId, status, error) => {
        set(state => {
          const doc = state.documents.get(documentId);
          if (doc) {
            doc.status = status;
            if (error) {
              doc.error_message = error;
            }
            if (status === 'indexed') {
              doc.indexed_at = new Date().toISOString();
            }
            state.documents.set(documentId, { ...doc });

            // Add notification
            const notificationType = status === 'indexed' ? 'success' : status === 'error' ? 'error' : 'info';
            const message = status === 'indexed'
              ? 'Document has been successfully indexed and is ready for use'
              : status === 'error'
              ? `Document processing failed: ${error || 'Unknown error'}`
              : 'Document status updated';

            get().addNotification({
              type: notificationType,
              documentId,
              documentName: doc.name,
              message
            });

            // Remove from upload progress if completed
            if (status === 'indexed' || status === 'error') {
              const progressEntry = Array.from(state.uploadProgress.values()).find(p => p.documentId === documentId);
              if (progressEntry) {
                state.uploadProgress.delete(progressEntry.documentId);
              }
            }
          }
          return {
            documents: new Map(state.documents),
            uploadProgress: new Map(state.uploadProgress)
          };
        });
      }
    }),
    {
      name: 'document-store',
      partialize: (state) => ({
        notifications: state.notifications.slice(0, 10), // Only persist recent notifications
        unreadNotificationCount: state.unreadNotificationCount
      })
    }
  )
);
