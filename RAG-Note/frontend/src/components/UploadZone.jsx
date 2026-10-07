import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

export default function UploadZone({ onUpload, loading }) {
  const onDrop = useCallback(
    (files) => {
      if (files.length) onUpload(files);
    },
    [onUpload]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    multiple: true,
    disabled: loading,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
      'image/*': ['.png', '.jpg', '.jpeg', '.webp', '.bmp'],
    },
  });

  const state = isDragReject
    ? 'border-rose-500 bg-rose-500/10'
    : isDragActive
    ? 'border-indigo-400 bg-indigo-500/10'
    : 'border-slate-700 hover:border-indigo-500/60 hover:bg-slate-800/60';

  return (
    <div
      {...getRootProps()}
      className={`cursor-pointer rounded-2xl border-2 border-dashed px-4 py-6 text-center transition ${state} ${
        loading ? 'pointer-events-none opacity-70' : ''
      }`}
    >
      <input {...getInputProps()} />
      {loading ? (
        <div className="flex flex-col items-center gap-3">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-600 border-t-indigo-400" />
          <p className="text-sm text-slate-300">Reading & indexing…</p>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-2">
          <svg className="h-8 w-8 text-indigo-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <p className="text-sm text-slate-200">
            <span className="font-semibold text-indigo-300">Click to upload</span> or drop files
          </p>
          <p className="text-xs text-slate-500">PDF · DOCX · TXT · MD · Images</p>
        </div>
      )}
    </div>
  );
}
