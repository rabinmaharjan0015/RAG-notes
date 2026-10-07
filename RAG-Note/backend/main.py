"""RAG Note — FastAPI entry point.  Run:  python3 main.py"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from rag.config import EMBEDDING_MODEL, QA_MODEL, VLM_MODEL, ENABLE_VLM, MEDIA_DIR, logger
from rag.model_loader import get_embedder, get_generator
from rag.routes import api_router
from rag.store import load_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"Generator:       {QA_MODEL or 'off (grounded extractive answers)'}")
    logger.info(f"Vision model:    {VLM_MODEL if ENABLE_VLM else 'off'} (loaded on first image)")
    await asyncio.to_thread(get_embedder)
    if QA_MODEL:
        await asyncio.to_thread(get_generator)
    await asyncio.to_thread(load_store)
    yield


app = FastAPI(title="RAG Note", version="5.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")


@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal error"})


if __name__ == "__main__":
    import uvicorn

    # No reload: it would restart (and reload models) whenever any file changes.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
