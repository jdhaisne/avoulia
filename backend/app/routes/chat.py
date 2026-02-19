import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.models import ChatRequest, ChatResponse
from app.rag import chat_simple, chat_simple_stream, stream_prompt
from app.haystack_rag import query_rag_haystack, get_rag_prompt_and_sources

router = APIRouter(prefix="/chat", tags=["chat"])


def _sse_line(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Envoie un message au chatbot.
    Si USE_RAG=true : RAG Haystack (Chroma). Sinon : chat simple (LLM seul).
    """
    settings = get_settings()
    history = [{"role": m.role, "content": m.content} for m in request.history]
    try:
        if settings.use_rag:
            answer, sources = query_rag_haystack(request.message, history)
            return ChatResponse(answer=answer, sources=sources)
        answer = chat_simple(request.message, history)
        return ChatResponse(answer=answer, sources=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur chat: {str(e)}")


def _stream_chat(request: ChatRequest):
    settings = get_settings()
    history = [{"role": m.role, "content": m.content} for m in request.history]
    try:
        if settings.use_rag:
            prompt_text, sources = get_rag_prompt_and_sources(request.message, history)
            for chunk in stream_prompt(prompt_text):
                yield _sse_line({"t": chunk})
            yield _sse_line({"done": True, "sources": sources})
        else:
            for chunk in chat_simple_stream(request.message, history):
                yield _sse_line({"t": chunk})
            yield _sse_line({"done": True, "sources": []})
    except Exception as e:
        yield _sse_line({"error": str(e)})


@router.post("/stream")
def chat_stream(request: ChatRequest):
    """
    Envoie un message et stream la réponse (SSE). Chaque événement : data: {"t": "fragment"}.
    Fin : data: {"done": true, "sources": [...]}.
    """
    return StreamingResponse(
        _stream_chat(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
