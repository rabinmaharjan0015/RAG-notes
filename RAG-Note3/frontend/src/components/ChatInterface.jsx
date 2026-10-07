import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const api = axios.create({ baseURL: '/api' });

const SUGGESTIONS = [
  'यो कागजातको मुख्य विषय के हो?',
  'Summarize the key points',
  'Show the table with the main figures',
];

const MODES = [
  { id: 'hybrid', label: 'Hybrid', hint: 'Vectors + knowledge graph (recommended)' },
  { id: 'local', label: 'Local', hint: 'Focus on the entities named in your question' },
  { id: 'global', label: 'Global', hint: 'Follow related topics across documents' },
  { id: 'naive', label: 'Naive', hint: 'Plain vector search only' },
];

const TYPE_ICON = { text: '📝', image: '🖼️', table: '📊', equation: '∑' };

function Md({ children }) {
  return (
    <div className="prose-chat">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
    </div>
  );
}

function Evidence({ e }) {
  return (
    <div className="rounded-lg border border-slate-700/60 bg-slate-900/70 p-2.5">
      <div className="mb-1.5 flex items-center justify-between gap-2 text-[11px] text-slate-500">
        <span className="truncate">
          {TYPE_ICON[e.type] || '📄'} {e.filename}
          {e.page ? ` · p.${e.page}` : ''}
          {e.section ? ` · ${e.section}` : ''}
        </span>
        <span className="shrink-0">
          {e.via !== 'vector' && <span className="mr-1 rounded bg-violet-500/20 px-1 text-violet-300">{e.via}</span>}
          match {(e.score * 100).toFixed(0)}%
        </span>
      </div>
      {e.type === 'image' && e.media_url && (
        <img src={e.media_url} alt={e.caption || 'figure'} className="mb-1.5 max-h-56 rounded-md border border-slate-700" />
      )}
      {e.type === 'table' && e.raw ? (
        <Md>{e.raw}</Md>
      ) : e.type === 'equation' && e.raw ? (
        <pre className="overflow-x-auto rounded bg-slate-950 p-2 text-xs text-emerald-200">{e.raw}</pre>
      ) : (
        <p className="text-xs leading-relaxed text-slate-300">{e.caption || e.snippet}…</p>
      )}
    </div>
  );
}

function Sources({ evidence, sources }) {
  const [open, setOpen] = useState(false);
  if (!sources?.length) return null;
  return (
    <div className="mt-3 border-t border-slate-700/60 pt-2">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 text-xs text-slate-400 transition hover:text-slate-200"
      >
        <span>{open ? '▾' : '▸'}</span>
        <span>
          Sources: {sources.join(', ')} · {evidence?.length || 0} item{evidence?.length === 1 ? '' : 's'}
        </span>
      </button>
      {open && <div className="mt-2 space-y-2">{evidence?.map((e, i) => <Evidence key={i} e={e} />)}</div>}
    </div>
  );
}

function Options({ options, onPick, disabled }) {
  if (!options?.length) return null;
  return (
    <div className="mt-3 flex flex-col gap-2">
      {options.map((o, i) => (
        <button
          key={i}
          disabled={disabled}
          onClick={() => onPick(o)}
          className="rounded-xl border border-indigo-500/40 bg-indigo-500/10 px-3 py-2 text-left text-xs leading-snug text-indigo-100 transition hover:border-indigo-400 hover:bg-indigo-500/20 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function Message({ msg, onPick, loading }) {
  const isUser = msg.role === 'user';
  const isClarify = msg.status === 'clarify';
  const figures = !isUser && msg.found !== false ? (msg.evidence || []).filter((e) => e.type === 'image' && e.media_url).slice(0, 2) : [];

  return (
    <div className={`flex animate-fade-up gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm ${
          isUser ? 'bg-indigo-500' : msg.isError ? 'bg-rose-500/20' : isClarify ? 'bg-indigo-500/20' : 'bg-slate-800'
        }`}
      >
        {isUser ? '🧑' : msg.isError ? '⚠️' : isClarify ? '🤔' : '🤖'}
      </div>
      <div
        className={`max-w-[88%] rounded-2xl px-4 py-3 text-sm sm:max-w-[78%] ${
          isUser
            ? 'rounded-tr-sm bg-indigo-600 text-white'
            : msg.isError
            ? 'rounded-tl-sm border border-rose-500/30 bg-rose-500/10 text-rose-200'
            : isClarify
            ? 'rounded-tl-sm border border-indigo-500/40 bg-indigo-500/10 text-indigo-50'
            : msg.found === false
            ? 'rounded-tl-sm border border-amber-500/30 bg-amber-500/10 text-amber-100'
            : 'rounded-tl-sm border border-slate-800 bg-slate-900 text-slate-100'
        }`}
      >
        {isUser ? (
          <>
            {msg.attachment && <p className="mb-1 text-xs text-indigo-200">📎 {msg.attachment}</p>}
            <p className="whitespace-pre-wrap">{msg.content}</p>
          </>
        ) : (
          <Md>{msg.content}</Md>
        )}

        {figures.map((e, i) => (
          <img key={i} src={e.media_url} alt={e.caption || 'figure'} className="mt-3 max-h-72 rounded-lg border border-slate-700" />
        ))}

        {!isUser && msg.visual_notes?.length > 0 && (
          <div className="mt-3 space-y-2 rounded-xl border border-violet-500/30 bg-violet-500/10 p-3">
            <p className="text-xs font-semibold text-violet-200">👁 Vision model</p>
            {msg.visual_notes.map((v, i) => (
              <div key={i} className="flex gap-2">
                <img src={v.media_url} alt="" className="h-14 w-14 shrink-0 rounded object-cover" />
                <p className="text-xs leading-relaxed text-violet-100">{v.note}</p>
              </div>
            ))}
          </div>
        )}

        {!isUser && msg.options?.length > 0 && <Options options={msg.options} onPick={onPick} disabled={loading} />}
        {!isUser && msg.found !== false && <Sources evidence={msg.evidence} sources={msg.sources} />}
      </div>
    </div>
  );
}

export default function ChatInterface({ documents, onNeedUpload }) {
  const [messages, setMessages] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('rag-note-chat') || '[]');
    } catch {
      return [];
    }
  });
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState([]);
  const [mode, setMode] = useState('hybrid');
  const [vlm, setVlm] = useState(false);
  const [attachment, setAttachment] = useState(null);
  const endRef = useRef(null);
  const taRef = useRef(null);
  const fileRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
    localStorage.setItem('rag-note-chat', JSON.stringify(messages.slice(-50)));
  }, [messages, loading]);

  useEffect(() => {
    setSelected((prev) => prev.filter((id) => documents.some((d) => d.id === id)));
  }, [documents]);

  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
  }, [input]);

  const send = async (text, opts = {}) => {
    const q = (text ?? input).trim();
    if (!q || loading) return;
    const file = opts.noAttachment ? null : attachment;
    setMessages((m) => [...m, { role: 'user', content: opts.display || q, attachment: file?.name }]);
    setInput('');
    setAttachment(null);
    setLoading(true);
    try {
      const ids = opts.documentIds?.length ? opts.documentIds : selected;
      let data;
      if (file) {
        const form = new FormData();
        form.append('question', q);
        form.append('mode', mode);
        form.append('vlm_enhanced', String(vlm));
        if (ids.length) form.append('document_ids', ids.join(','));
        form.append('image', file);
        ({ data } = await api.post('/ask-multimodal', form));
      } else {
        const payload = { question: q, mode, vlm_enhanced: vlm };
        if (ids.length) payload.document_ids = ids;
        ({ data } = await api.post('/ask', payload));
      }
      setMessages((m) => [...m, { role: 'assistant', ...data }]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: 'assistant', isError: true, content: err.response?.data?.detail || 'Something went wrong. Is the backend running?' },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const pickOption = (o) => send(o.query, { documentIds: o.document_ids, display: o.label, noAttachment: true });

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const toggle = (id) => setSelected((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));

  if (!documents.length) {
    return (
      <div className="flex flex-1 items-center justify-center p-6">
        <div className="max-w-md text-center">
          <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500/20 to-violet-500/20 text-3xl ring-1 ring-indigo-500/30">
            💬
          </div>
          <h2 className="text-xl font-semibold">Ask questions about your documents</h2>
          <p className="mt-2 text-sm text-slate-400">
            Upload PDFs, Word, text or Markdown files and images. Text, tables, figures and equations are all searchable —
            answers come directly from your documents, in Nepali or English.
          </p>
          <button
            onClick={onNeedUpload}
            className="mt-5 rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-medium text-white shadow-lg shadow-indigo-600/30 transition hover:bg-indigo-500 lg:hidden"
          >
            Upload a document
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="scroll-thin flex gap-2 overflow-x-auto border-b border-slate-800 px-4 py-2.5 sm:px-6">
        <button
          onClick={() => setSelected([])}
          className={`shrink-0 rounded-full border px-3 py-1 text-xs font-medium transition ${
            selected.length === 0 ? 'border-indigo-500 bg-indigo-500/15 text-indigo-200' : 'border-slate-700 text-slate-400 hover:border-slate-500'
          }`}
        >
          All documents
        </button>
        {documents.map((d) => (
          <button
            key={d.id}
            onClick={() => toggle(d.id)}
            title={d.filename}
            className={`shrink-0 rounded-full border px-3 py-1 text-xs font-medium transition ${
              selected.includes(d.id) ? 'border-indigo-500 bg-indigo-500/15 text-indigo-200' : 'border-slate-700 text-slate-400 hover:border-slate-500'
            }`}
          >
            📄 {d.filename.length > 22 ? d.filename.slice(0, 22) + '…' : d.filename}
          </button>
        ))}
      </div>

      <div className="scroll-thin flex-1 overflow-y-auto px-4 py-6 sm:px-6">
        <div className="mx-auto max-w-3xl space-y-5">
          {messages.length === 0 && (
            <div className="animate-fade-up py-10 text-center">
              <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-800 text-2xl">🤖</div>
              <p className="text-slate-300">Ask anything about your documents — text, tables, figures or formulas.</p>
              <div className="mt-5 flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="rounded-full border border-slate-700 bg-slate-900 px-4 py-2 text-xs text-slate-300 transition hover:border-indigo-500 hover:text-white"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) => (
            <Message key={i} msg={i === messages.length - 1 ? m : { ...m, options: undefined }} onPick={pickOption} loading={loading} />
          ))}

          {loading && (
            <div className="flex gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-slate-800 text-sm">🤖</div>
              <div className="flex items-center gap-1.5 rounded-2xl rounded-tl-sm border border-slate-800 bg-slate-900 px-4 py-3.5">
                {[0, 0.16, 0.32].map((d) => (
                  <span key={d} className="h-2 w-2 animate-bounce3 rounded-full bg-indigo-400" style={{ animationDelay: `${d}s` }} />
                ))}
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>
      </div>

      <div className="border-t border-slate-800 bg-slate-950 px-4 py-3 sm:px-6">
        {/* query controls */}
        <div className="mx-auto mb-2 flex max-w-3xl flex-wrap items-center gap-x-4 gap-y-2 text-xs text-slate-400">
          <label className="flex items-center gap-2" title={MODES.find((m) => m.id === mode)?.hint}>
            <span>Search mode</span>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              className="rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200 outline-none focus:border-indigo-500"
            >
              {MODES.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <label className="flex cursor-pointer items-center gap-2" title="Let the vision model look at retrieved figures (slower)">
            <input type="checkbox" checked={vlm} onChange={(e) => setVlm(e.target.checked)} className="accent-indigo-500" />
            <span>Analyse images</span>
          </label>
          {attachment && (
            <span className="flex items-center gap-1.5 rounded-full border border-indigo-500/40 bg-indigo-500/10 px-2.5 py-1 text-indigo-200">
              📎 {attachment.name}
              <button onClick={() => setAttachment(null)} className="text-indigo-300 hover:text-white" aria-label="Remove attachment">
                ✕
              </button>
            </span>
          )}
        </div>

        <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-2xl border border-slate-700 bg-slate-900 p-2 transition focus-within:border-indigo-500 focus-within:ring-2 focus-within:ring-indigo-500/20">
          <input
            ref={fileRef}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/bmp"
            className="hidden"
            onChange={(e) => {
              setAttachment(e.target.files?.[0] || null);
              e.target.value = '';
            }}
          />
          <button
            onClick={() => fileRef.current?.click()}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-slate-400 transition hover:bg-slate-800 hover:text-slate-100"
            title="Attach an image to ask about it"
            aria-label="Attach image"
          >
            <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21.4 11.1l-9.2 9.2a6 6 0 0 1-8.5-8.5l9.2-9.2a4 4 0 0 1 5.7 5.7l-9.2 9.2a2 2 0 0 1-2.8-2.8l8.5-8.5" />
            </svg>
          </button>
          <textarea
            ref={taRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={attachment ? 'Ask about the attached image…' : 'Ask a question… / प्रश्न सोध्नुहोस्…'}
            className="max-h-40 flex-1 resize-none bg-transparent px-2 py-2 text-sm text-slate-100 placeholder-slate-500 outline-none"
          />
          <button
            onClick={() => send()}
            disabled={loading || !input.trim()}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-indigo-600 text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-500"
            aria-label="Send"
          >
            <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
            </svg>
          </button>
        </div>
        <p className="mx-auto mt-2 max-w-3xl text-center text-[11px] text-slate-600">
          Answers come from your uploaded documents only. Enter to send · Shift+Enter for a new line.
        </p>
      </div>
    </div>
  );
}
