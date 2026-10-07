import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import Header from './components/Header';
import UploadZone from './components/UploadZone';
import DocumentList from './components/DocumentList';
import ChatInterface from './components/ChatInterface';

const api = axios.create({ baseURL: '/api' });

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [toast, setToast] = useState(null);
  const [chatKey, setChatKey] = useState(0);

  const notify = useCallback((message, type = 'info') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 9000);
  }, []);

  const fetchDocuments = useCallback(async () => {
    try {
      const res = await api.get('/documents');
      setDocuments(res.data.documents || []);
    } catch {
      notify('Cannot reach the backend. Is it running on port 8000?', 'error');
    }
  }, [notify]);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const handleUpload = async (files) => {
    setUploading(true);
    try {
      const form = new FormData();
      files.forEach((f) => form.append('files', f));
      const { data } = await api.post('/upload-multiple', form);
      await fetchDocuments();
      const fresh = data.documents.filter((d) => !d.duplicate).length;
      const dupes = data.documents.length - fresh;
      if (data.errors?.length) notify(data.errors.map((e) => `${e.filename}: ${e.error}`).join('\n'), 'error');
      else if (dupes && !fresh) notify('Already uploaded — nothing to process again.', 'info');
      else {
        const warn = data.documents.flatMap((d) => d.warnings || [])[0];
        notify(
          `${fresh} document${fresh !== 1 ? 's' : ''} added${dupes ? ` (${dupes} already existed)` : ''}.` + (warn ? `\n⚠ ${warn}` : ''),
          warn ? 'info' : 'success'
        );
      }
    } catch (err) {
      notify(err.response?.data?.detail || 'Upload failed. Please try again.', 'error');
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (id) => {
    try {
      await api.delete(`/documents/${id}`);
      setDocuments((prev) => prev.filter((d) => d.id !== id));
    } catch {
      notify('Could not delete the document.', 'error');
    }
  };

  const handleDeleteAll = async () => {
    if (!documents.length || !window.confirm('Delete all documents?')) return;
    try {
      await api.delete('/documents');
      setDocuments([]);
    } catch {
      notify('Could not delete documents.', 'error');
    }
  };

  const toastStyle = {
    success: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200',
    error: 'border-rose-500/40 bg-rose-500/10 text-rose-200',
    info: 'border-slate-600 bg-slate-800 text-slate-200',
  };

  return (
    <div className="flex h-full bg-slate-950">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-30 bg-black/60 lg:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-80 max-w-[85vw] flex-col border-r border-slate-800 bg-slate-900 transition-transform duration-200 lg:static lg:translate-x-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex items-center gap-3 border-b border-slate-800 px-5 py-4">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 text-lg shadow-lg shadow-indigo-500/30">
            📄
          </div>
          <div>
            <h1 className="text-base font-semibold leading-tight">RAG Note</h1>
            <p className="text-xs text-slate-400">Ask your documents</p>
          </div>
        </div>

        <div className="p-4">
          <UploadZone onUpload={handleUpload} loading={uploading} />
        </div>

        <div className="flex items-center justify-between px-5 pb-2">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Documents <span className="ml-1 rounded-full bg-slate-800 px-2 py-0.5 text-slate-300">{documents.length}</span>
          </h2>
          {documents.length > 0 && (
            <button onClick={handleDeleteAll} className="text-xs text-slate-500 transition hover:text-rose-400">
              Delete all
            </button>
          )}
        </div>

        <div className="scroll-thin flex-1 overflow-y-auto px-4 pb-4">
          <DocumentList documents={documents} onDelete={handleDelete} />
        </div>
      </aside>

      {/* Main */}
      <main className="flex min-w-0 flex-1 flex-col">
        <Header
          docCount={documents.length}
          onMenu={() => setSidebarOpen(true)}
          onClear={() => setChatKey((k) => k + 1)}
        />
        <ChatInterface key={chatKey} documents={documents} onNeedUpload={() => setSidebarOpen(true)} />
      </main>

      {toast && (
        <div
          className={`fixed bottom-5 right-5 z-50 max-w-sm animate-fade-up whitespace-pre-line rounded-xl border px-4 py-3 text-sm shadow-xl ${toastStyle[toast.type]}`}
        >
          {toast.message}
        </div>
      )}
    </div>
  );
}
