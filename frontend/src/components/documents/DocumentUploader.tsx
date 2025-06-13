import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, File as FileIcon } from 'lucide-react';
import { useDocumentStore } from '@/stores/documentStore';
import { Button } from '@/components/common';
import { cn, formatFileSize } from '@/utils';

interface DocumentUploaderProps {
  projectId: string;
  onClose?: () => void;
}

export const DocumentUploader: React.FC<DocumentUploaderProps> = ({
  projectId,
  onClose
}) => {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const { uploadDocument, uploadProgress } = useDocumentStore();

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    setFiles(prev => [...prev, ...acceptedFiles]);

    // Handle rejected files
    const newErrors: Record<string, string> = {};
    rejectedFiles.forEach((file: any) => {
      const error = file.errors[0];
      if (error.code === 'file-too-large') {
        newErrors[file.file.name] = 'File size exceeds 50MB limit';
      } else if (error.code === 'file-invalid-type') {
        newErrors[file.file.name] = 'File type not supported';
      } else {
        newErrors[file.file.name] = error.message;
      }
    });
    setErrors(prev => ({ ...prev, ...newErrors }));
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
      'text/csv': ['.csv']
    },
    maxSize: 50 * 1024 * 1024, // 50MB
    multiple: true
  });

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (files.length === 0) return;

    setUploading(true);
    const uploadErrors: Record<string, string> = {};

    for (const file of files) {
      try {
        await uploadDocument(projectId, file);
      } catch (error) {
        uploadErrors[file.name] = error instanceof Error ? error.message : 'Upload failed';
      }
    }

    setUploading(false);

    if (Object.keys(uploadErrors).length === 0) {
      setFiles([]);
      onClose?.();
    } else {
      setErrors(uploadErrors);
    }
  };

  const currentUploads = Array.from(uploadProgress.values());

  return (
    <div className="space-y-4">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={cn(
          'border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer',
          isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
        )}
      >
        <input {...getInputProps()} />
        <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        {isDragActive ? (
          <p className="text-blue-600">Drop the files here...</p>
        ) : (
          <>
            <p className="text-gray-700 mb-2">
              Drag & drop files here, or click to select
            </p>
            <p className="text-sm text-gray-500">
              Supports PDF, DOCX, TXT, MD, CSV (max 50MB per file)
            </p>
          </>
        )}
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="space-y-2">
          <h4 className="font-medium text-gray-900">Selected Files</h4>
          {files.map((file, index) => (
            <div
              key={`${file.name}-${index}`}
              className={cn(
                'flex items-center justify-between p-3 bg-gray-50 rounded-lg',
                errors[file.name] && 'bg-red-50'
              )}
            >
              <div className="flex items-center gap-3">
                <FileIcon className="w-5 h-5 text-gray-500" />
                <div>
                  <p className="font-medium text-gray-900">{file.name}</p>
                  <p className="text-sm text-gray-500">{formatFileSize(file.size)}</p>
                  {errors[file.name] && (
                    <p className="text-sm text-red-600 mt-1">{errors[file.name]}</p>
                  )}
                </div>
              </div>
              <button
                onClick={() => removeFile(index)}
                className="p-1 hover:bg-gray-200 rounded"
                disabled={uploading}
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Upload Progress */}
      {currentUploads.length > 0 && (
        <div className="space-y-2">
          <h4 className="font-medium text-gray-900">Upload Progress</h4>
          {currentUploads.map((upload) => (
            <div key={upload.documentId} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-gray-700">{upload.fileName}</span>
                <span className="text-gray-500">
                  {upload.status === 'uploading' && `${upload.progress}%`}
                  {upload.status === 'processing' && 'Processing...'}
                  {upload.status === 'error' && 'Failed'}
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={cn(
                    'h-2 rounded-full transition-all',
                    upload.status === 'error' ? 'bg-red-500' : 'bg-blue-500'
                  )}
                  style={{ width: `${upload.progress}%` }}
                />
              </div>
              {upload.error && (
                <p className="text-xs text-red-600">{upload.error}</p>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Actions */}
      <div className="flex justify-end gap-3">
        <Button variant="secondary" onClick={onClose} disabled={uploading}>
          Cancel
        </Button>
        <Button
          onClick={handleUpload}
          disabled={files.length === 0 || uploading}
          isLoading={uploading}
        >
          {uploading ? 'Uploading...' : `Upload ${files.length} File${files.length !== 1 ? 's' : ''}`}
        </Button>
      </div>
    </div>
  );
};
