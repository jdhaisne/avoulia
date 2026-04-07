import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.models import ChatRequest, ChatResponse, SuggestedCase
from app.rag import chat_simple, chat_simple_stream, stream_prompt
from app.haystack_rag import query_rag_haystack, get_rag_prompt_and_sources, WELCOME_MESSAGE

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("/welcome")
def chat_welcome():
    """Retourne le premier message que l'agent affiche au chargement du chat."""
    return {"message": WELCOME_MESSAGE}


def _sse_line(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _last_suggested_cases_to_dicts(request: ChatRequest) -> list[dict] | None:
    if not request.last_suggested_cases:
        return None
    out: list[dict] = []
    for c in request.last_suggested_cases:
        d: dict = {"id": c.id, "content": c.content}
        if c.effort is not None:
            d["effort"] = c.effort
        if c.prerequis_donnees is not None:
            d["prerequis_donnees"] = c.prerequis_donnees
        if c.guardrails is not None:
            d["guardrails"] = c.guardrails
        if c.questions_qualification is not None:
            d["questions_qualification"] = c.questions_qualification
        if c.sensibilite_donnees is not None:
            d["sensibilite_donnees"] = c.sensibilite_donnees
        out.append(d)
    return out


def _build_suggested_cases(
    ids: list[str],
    full_contents: list[str],
    case_extras: list[dict[str, str | None]] | None = None,
) -> list[SuggestedCase]:
    n = min(len(ids), len(full_contents))
    rows: list[SuggestedCase] = []
    for i in range(n):
        ex = case_extras[i] if case_extras and i < len(case_extras) else {}
        rows.append(
            SuggestedCase(
                id=ids[i],
                content=full_contents[i],
                effort=ex.get("effort"),
                prerequis_donnees=ex.get("prerequis_donnees"),
                guardrails=ex.get("guardrails"),
                questions_qualification=ex.get("questions_qualification"),
                sensibilite_donnees=ex.get("sensibilite_donnees"),
            )
        )
    return rows


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Envoie un message au chatbot.
    « ok / vas-y / oui » exécute l'action en attente (ex. détailler le cas) si pending_action est renvoyé.
    """
    settings = get_settings()
    history = [{"role": m.role, "content": m.content} for m in request.history]
    try:
        if settings.use_rag:
            last = _last_suggested_cases_to_dicts(request)
            (
                answer,
                sources,
                suggested_case_ids,
                full_contents,
                case_extras,
                pending_action,
                pending_use_case_id,
                pending_case_index,
                selected_domain_code,
                selected_sector,
                selected_intention,
            ) = query_rag_haystack(
                request.message,
                history,
                last_suggested_cases=last,
                pending_action=request.pending_action,
                pending_use_case_id=request.pending_use_case_id,
                selected_domain_code=request.selected_domain_code,
                selected_sector=request.selected_sector,
                selected_intention=request.selected_intention,
            )
            suggested_cases = (
                _build_suggested_cases(suggested_case_ids, full_contents, case_extras) if full_contents else None
            )
            return ChatResponse(
                answer=answer,
                sources=sources,
                suggested_case_ids=suggested_case_ids,
                suggested_cases=suggested_cases,
                pending_action=pending_action,
                pending_use_case_id=pending_use_case_id,
                pending_case_index=pending_case_index,
                selected_domain_code=selected_domain_code,
                selected_sector=selected_sector,
                selected_intention=selected_intention,
            )
        answer = chat_simple(request.message, history)
        return ChatResponse(answer=answer, sources=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur chat: {str(e)}")


def _stream_chat(request: ChatRequest):
    settings = get_settings()
    history = [{"role": m.role, "content": m.content} for m in request.history]
    try:
        if settings.use_rag:
            last = _last_suggested_cases_to_dicts(request)
            (
                prompt_text,
                sources,
                suggested_case_ids,
                full_contents,
                case_extras,
                selected_domain_code,
                selected_sector,
                selected_intention,
            ) = get_rag_prompt_and_sources(
                request.message,
                history,
                last_suggested_cases=last,
                pending_action=request.pending_action,
                pending_use_case_id=request.pending_use_case_id,
                selected_domain_code=request.selected_domain_code,
                selected_sector=request.selected_sector,
                selected_intention=request.selected_intention,
            )
            for chunk in stream_prompt(prompt_text):
                yield _sse_line({"t": chunk})
            suggested_cases = _build_suggested_cases(suggested_case_ids, full_contents, case_extras)
            done_payload = {
                "done": True,
                "sources": sources,
                "suggested_case_ids": suggested_case_ids,
                "suggested_cases": [c.model_dump(exclude_none=True) for c in suggested_cases],
                "selected_domain_code": selected_domain_code,
                "selected_sector": selected_sector,
                "selected_intention": selected_intention,
            }
            yield _sse_line(done_payload)
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
