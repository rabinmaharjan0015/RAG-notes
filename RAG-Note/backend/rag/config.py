"""Central configuration: paths, limits and environment settings."""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag-note")

# ─── Paths ──────────────────────────────────────────────────────
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
EMB_DIR = DATA_DIR / "embeddings"
MEDIA_DIR = DATA_DIR / "media"          # extracted figures, served at /media
STORE_FILE = DATA_DIR / "store.json"

for _d in (UPLOAD_DIR, DATA_DIR, MODELS_DIR, EMB_DIR, MEDIA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── Limits ─────────────────────────────────────────────────────
MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_DOCS = 200
DOC_EXTS = {".txt", ".pdf", ".md", ".docx"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALLOWED_EXTS = DOC_EXTS | IMAGE_EXTS

# ─── Chunking ───────────────────────────────────────────────────
CHUNK_CHARS = 700
CHUNK_OVERLAP = 120

# ─── Models ─────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
# Optional generator (OFF by default). Example: Qwen/Qwen2.5-1.5B-Instruct
QA_MODEL = os.getenv("HF_QA_MODEL", "").strip()
# Vision-language model: describes figures/charts/photos. Loaded lazily (only when the
# first image is met), downloaded once. Set ENABLE_VLM=false to skip images' AI captions.
VLM_MODEL = os.getenv("HF_VLM_MODEL", "HuggingFaceTB/SmolVLM-256M-Instruct").strip()
ENABLE_VLM = os.getenv("ENABLE_VLM", "true").strip().lower() in ("1", "true", "yes")
MAX_VLM_IMAGES_PER_DOC = int(os.getenv("MAX_VLM_IMAGES_PER_DOC", "25"))

# ─── Multimodal parsing ─────────────────────────────────────────
MIN_IMAGE_PT = 40            # ignore icons / bullets smaller than this (PDF points)
CONTEXT_CHARS = 300          # surrounding text attached to figures/tables/equations
TABLE_ROWS_PER_ITEM = 8      # long tables are split into row groups (header repeated)
ENTITIES_PER_ITEM = 8        # knowledge-graph entities kept per item
RRF_K = 60                   # reciprocal-rank-fusion constant for hybrid retrieval
MODALITY_BOOST = 0.06        # ranking boost when the question asks for a table/figure/equation

# ─── Retrieval & answer decision ────────────────────────────────
# Embedding scores differ a lot between models, so decisions use RELATIVE signals
# (how much the best chunk stands out from all other chunks) plus keyword evidence,
# instead of one absolute cosine threshold. All values can be tuned in .env.
DENSE_WEIGHT = 0.8
LEXICAL_WEIGHT = 0.2
CANDIDATES = 50                                   # chunks re-ranked with keyword evidence
Z_STRONG = float(os.getenv("Z_STRONG", "3.0"))    # best chunk this far above average => answer
Z_MIN = float(os.getenv("Z_MIN", "1.5"))          # minimum when keywords also match
MIN_KEYWORD_COVERAGE = float(os.getenv("MIN_KEYWORD_COVERAGE", "0.5"))
SMALL_CORPUS_CHUNKS = 10                          # below this z-scores are unreliable
AMBIG_RATIO = float(os.getenv("AMBIG_RATIO", "0.80"))     # within 80% of best score = "equally good"
AMBIG_MAX_TERMS = 2                               # only short questions can be ambiguous
MAX_OPTIONS = 4
