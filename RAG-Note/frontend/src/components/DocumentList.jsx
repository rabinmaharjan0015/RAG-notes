import React from 'react';

const ICONS = { pdf: '📕', docx: '📘', txt: '📄', md: '📝', png: '🖼️', jpg: '🖼️', jpeg: '🖼️', webp: '🖼️', bmp: '🖼️' };

function fmtSize(b) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
}

export default function DocumentList({ documents, onDelete }) {
  if (!documents.length) {
    return (
      <div className="mt-6 rounded-xl border border-slate-800 bg-slate-900/60 p-6 text-center">
        <div className="text-3xl">📂</div>
        <p className="mt-2 text-sm text-slate-400">Upload a document to get started.</p>
      </div>
    );
  }

  return (
    <ul className="space-y-2">
      {documents.map((doc) => {
        const ext = doc.filename.split('.').pop().toLowerCase();
        return (
          <li
            key={doc.id}
            className="group flex items-start gap-3 rounded-xl border border-slate-800 bg-slate-900/60 p-3 transition hover:border-slate-700 hover:bg-slate-800/60"
          >
            <span className="mt-0.5 text-xl">{ICONS[ext] || '📁'}</span>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-slate-100" title={doc.filename}>
                {doc.filename}
              </p>
              <p className="mt-0.5 text-xs text-slate-500">
                {fmtSize(doc.size)}
                {doc.pages > 1 && ` · ${doc.pages} pages`}
                {doc.is_nepali && <span className="ml-1 rounded bg-indigo-500/15 px-1.5 py-0.5 text-indigo-300">नेपाली</span>}
              </p>
              <div className="mt-1.5 flex flex-wrap gap-1">
                {[
                  ['text', '📝', 'text'],
                  ['image', '🖼️', 'images'],
                  ['table', '📊', 'tables'],
                  ['equation', '∑', 'equations'],
                ].map(([k, icon, label]) =>
                  doc.item_counts?.[k] ? (
                    <span key={k} title={`${doc.item_counts[k]} ${label}`} className="rounded-md bg-slate-800 px-1.5 py-0.5 text-[11px] text-slate-300">
                      {icon} {doc.item_counts[k]}
                    </span>
                  ) : null
                )}
              </div>
            </div>
            <button
              onClick={() => onDelete(doc.id)}
              className="rounded-md p-1.5 text-slate-500 opacity-0 transition hover:bg-rose-500/10 hover:text-rose-400 focus:opacity-100 group-hover:opacity-100"
              title="Delete"
              aria-label={`Delete ${doc.filename}`}
            >
              <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" />
              </svg>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
