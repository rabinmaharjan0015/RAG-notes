# RAG Note — multimodal RAG with Hugging Face models 📄🖼️📊∑

A RAG-Anything-style document Q&A system that runs **fully locally on Hugging Face models**
(no OpenAI key). It indexes **text, tables, figures and equations** from PDF / DOCX / MD / TXT /
images, in **Nepali, English or Hindi**, and answers only from your documents.

## RAG-Anything → RAG Note

| RAG-Anything | RAG Note |
|---|---|
| Parsing (MinerU / Docling / PaddleOCR) | **pdfplumber** (PDF), python-docx (DOCX), Markdown/TXT parser, Pillow (images). Output uses RAG-Anything's `content_list` format |
| Image / table / equation processors | `modal_processors.py` — figures described by a **Hugging Face VLM** (SmolVLM by default), tables split into searchable row groups, equations kept as formulas; each linked to its surrounding text (context-aware) |
| Multimodal knowledge graph | `graph.py` — entities (key terms), entity co-occurrence links, and figure/table ↔ text links. Built statistically, **no LLM** |
| Modes `naive / local / global / hybrid` | Same four modes, fused with Reciprocal Rank Fusion; modality-aware ranking (asking for a "table" boosts tables) |
| VLM-enhanced query | "Analyse images" toggle — the VLM looks at retrieved figures |
| Multimodal query (`aquery_with_multimodal`) | Attach an image (UI) or send a table / LaTeX equation (`/api/ask-multimodal`) |
| Direct content-list insertion | `POST /api/insert-content-list` |
| — | **Accuracy gate**: asks a clarifying question for vague questions, says "not found" instead of guessing |

Embeddings: `intfloat/multilingual-e5-small`. Every model is downloaded **once** into `backend/models/`
and then loaded offline.

## Run

```bash
./start.sh
```

First start downloads the embedding model once. The vision model downloads the first time you
index an image. Later starts are offline.

Manual:

```bash
cd backend && pip install -r requirements.txt && cp .env.example .env && python3 main.py
cd frontend && npm install && npm run dev      # http://localhost:3000
```

## How a question is answered

| Situation | What happens |
|---|---|
| Generic question ("tell me about this", "यो के हो?") | Asks what you mean; offers topics from your documents |
| "Summarize" with several documents | Asks which document |
| Short term matching several different documents/sections ("fee") | Asks which one you mean |
| Formula in the question (`F = m a`) | Exact match against indexed equations |
| Nothing in the documents supports it | "Not found" + topics your documents cover |
| Supported question | **Text** → sentences from the document · **Table** → the table · **Figure** → the image + caption · **Equation** → the formula, with sources |

A question is answered only when the documents contain real evidence:
(A) the question's own words appear in the best items **and** they stand out a little (`Z_MIN`), or
(B) some of its words appear and the item stands out clearly (`Z_STRONG`), or
(C) no words are shared (e.g. a question in another language) but the item stands out hugely (`Z_CROSS`).
Ranking fuses meaning and keyword matches (Reciprocal Rank Fusion, so it does not depend on the
embedding model's score scale). The knowledge graph adds context but never displaces the top
evidence-backed results and never makes an unsupported question look answerable.

### PDFs with a damaged Nepali text layer

Some PDFs (typically web pages "printed to PDF") lose Nepali conjuncts in their text layer: कार्यालय
extracts as कायालय, लेखापरीक्षण as लेखापरीण. This is damage in the file itself — every extractor reads it
the same way. RAG Note handles it in three steps:

1. **Detects it** per page (healthy Nepali has ~5–13 % halants per consonant; damaged pages have ~0 %, or contain unmapped glyphs).
2. **OCR fallback**: if Tesseract with Nepali data is installed, those pages are re-read from the rendered page image (`OCR_ENABLED`, `OCR_LANGS`).
3. **Without OCR**: the document is flagged ⚠ in the UI, and word matching tolerates the missing letters
   (so a correctly spelled question still finds लेखापरीणको). Answers then quote the damaged spelling.

## Search modes

| Mode | Use for |
|---|---|
| `hybrid` (default) | vectors + local + global graph search |
| `local` | specific facts about named things |
| `global` | broad themes / connected topics |
| `naive` | plain vector search |

## Tuning (`backend/.env`)

- `Z_STRONG`, `Z_MIN`, `MIN_KEYWORD_COVERAGE`, `AMBIG_RATIO` — accuracy gate (see `.env.example`)
- `HF_VLM_MODEL` / `ENABLE_VLM` — figure understanding; `HuggingFaceTB/SmolVLM-500M-Instruct` is better, slower
- `HF_EMBEDDING_MODEL` — `intfloat/multilingual-e5-base` is more accurate
- `HF_QA_MODEL` — optional LLM that phrases the answer from retrieved text only

Debug why something was answered/refused:

```bash
curl -s localhost:8000/api/ask -H 'Content-Type: application/json' \
  -d '{"question":"your question","mode":"hybrid","debug":true}' | python3 -m json.tool
```

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/upload-multiple` | Upload PDF, DOCX, TXT, MD, PNG/JPG/WebP/BMP |
| GET | `/api/documents` | List documents (with text/image/table/equation counts) |
| GET | `/api/documents/{id}/items` | Every indexed item of a document |
| DELETE | `/api/documents/{id}` · `/api/documents` | Delete one / all |
| POST | `/api/ask` | `{question, mode, document_ids?, vlm_enhanced?, debug?}` |
| POST | `/api/ask-multimodal` | form: `question`, `image?`, `table_data?` (markdown/CSV), `latex?`, `mode` |
| POST | `/api/insert-content-list` | `{file_name, content_list:[{type:"text"\|"image"\|"table"\|"equation", ...}]}` |
| GET | `/api/graph` | Knowledge-graph summary (entities, links, top entities) |
| GET | `/api/capabilities` · `/api/stats` | Models in use, index statistics |

## Known limits (honest)

- **Parser**: pdfplumber is lighter than MinerU. It handles digital PDFs well but not multi-column
  layouts or vector-drawn charts (only embedded images are extracted). Tables with merged header
  cells come out untidy (numbers stay correct). PDFs using legacy Nepali fonts (Preeti) extract as
  garbled text; scanned PDFs need OCR. The Nepali OCR path has not been verified with real Nepali
  language data — check the OCR'd text of your own files.
- **Equations** are detected heuristically and stored as text, not true LaTeX (Word equations and `$$…$$`
  in Markdown are exact).
- **Graph** entities are statistical key terms, not LLM-extracted entities/relations as in LightRAG.
- **Small VLM on CPU**: SmolVLM-256M reads simple charts and photos; it is weak at dense text and
  Nepali. Indexing images takes a few seconds each.

## Structure

```
RAG-Note/
├── start.sh
├── backend/
│   ├── main.py                 # FastAPI app (python3 main.py)
│   └── rag/
│       ├── config.py           # paths, limits, environment settings
│       ├── schemas.py          # request / response models
│       ├── model_loader.py     # download-once HF models: embeddings, VLM, optional LLM
│       ├── parsers/            # pdf.py, docx_parser.py, plain.py, image.py, common.py
│       ├── modal_processors.py # image / table / equation -> searchable items
│       ├── graph.py            # multimodal knowledge graph (entities + links)
│       ├── store.py            # persistence (JSON + embeddings + media)
│       ├── ingest.py           # parse -> items -> embeddings -> graph
│       ├── retrieval.py        # naive / local / global / hybrid retrieval
│       ├── clarify.py          # generic / ambiguous / summary clarifying questions
│       ├── qa.py               # decision flow + answer building
│       ├── text_utils.py       # chunking, tokenising, stemming, question analysis
│       └── routes/             # system, documents, ask, multimodal
└── frontend/                   # React + Vite + Tailwind
    └── src/components/         # Header, UploadZone, DocumentList, ChatInterface
```
