from fastapi import APIRouter

from . import system, documents, ask, multimodal

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(documents.router)
api_router.include_router(ask.router)
api_router.include_router(multimodal.router)
