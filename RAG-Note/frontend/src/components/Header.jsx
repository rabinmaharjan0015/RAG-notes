import React from 'react';

export default function Header({ docCount, onMenu, onClear }) {
  return (
    <header className="flex items-center justify-between border-b border-slate-800 bg-slate-950/80 px-4 py-3 backdrop-blur sm:px-6">
      <div className="flex items-center gap-3">
        <button
          onClick={onMenu}
          className="rounded-lg p-2 text-slate-300 transition hover:bg-slate-800 lg:hidden"
          aria-label="Open documents"
        >
          <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
        <div>
          <h2 className="text-sm font-semibold sm:text-base">Chat with your documents</h2>
          <p className="text-xs text-slate-400">
            {docCount ? `Searching ${docCount} document${docCount !== 1 ? 's' : ''}` : 'No documents yet'} · नेपाली / English
          </p>
        </div>
      </div>
      <button
        onClick={onClear}
        className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:border-slate-500 hover:bg-slate-800"
      >
        New chat
      </button>
    </header>
  );
}
