import asyncio

from fastapi import APIRouter, HTTPException

from ..qa import answer_question
from ..schemas import QuestionRequest, AnswerResponse
from ..store import documents_store

router = APIRouter(prefix="/api")


@router.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    if not documents_store:
        raise HTTPException(400, "No documents uploaded yet.")
    return await asyncio.to_thread(answer_question, req)
