"""Pydantic request / response models."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    document_ids: Optional[List[str]] = None
    max_sources: int = Field(default=5, ge=1, le=20)
    mode: str = "hybrid"            # naive (vectors) | local (entities) | global (relations) | hybrid (all)
    vlm_enhanced: bool = False      # let the vision model look at retrieved images
    debug: bool = False


class Source(BaseModel):
    filename: str
    snippet: str
    score: float
    type: str = "text"              # text | image | table | equation
    page: Optional[int] = None      # 1-based page (PDF only)
    section: Optional[str] = None
    media_url: Optional[str] = None  # figure image
    raw: Optional[str] = None       # markdown table / LaTeX
    caption: Optional[str] = None
    via: str = "vector"             # vector | graph | structure


class Option(BaseModel):
    label: str                                  # text shown on the button
    query: str                                  # question sent when clicked
    document_ids: Optional[List[str]] = None    # optionally narrow to one document


class VisualNote(BaseModel):
    media_url: str
    note: str


class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]
    source_count: int
    confidence: float = 0.0
    found: bool = True
    status: str = "answer"                      # "answer" | "clarify" | "not_found"
    options: List[Option] = []
    evidence: List[Source] = []
    visual_notes: List[VisualNote] = []
    mode: str = "hybrid"
    diagnostics: Optional[dict] = None


class DocumentInfo(BaseModel):
    id: str
    filename: str
    size: int
    content_preview: str
    uploaded_at: str = ""
    chunk_count: int = 0
    is_nepali: bool = False
    duplicate: bool = False
    item_counts: Dict[str, int] = {}            # {"text": 12, "image": 2, "table": 1, "equation": 0}
    pages: int = 0
    warnings: List[str] = []                    # e.g. damaged text layer, OCR used


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


class ContentListRequest(BaseModel):
    file_name: str = "content_list"
    content_list: List[Dict[str, Any]]
