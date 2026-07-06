from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models import Paper
from app.services.llm import LLMService
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore

router = APIRouter()

class CompareRequest(BaseModel):
    paper_ids: List[str]
    aspect: str = 'methodology, findings, and contributions'

COMPARE_SYSTEM = """You are a research analyst comparing multiple academic papers.
Provide a structured comparison table covering the requested aspects.
Be objective, technical, and highlight both similarities and differences.
Format: use clear sections for each comparison axis."""

@router.post('/')
async def compare_papers(
    body: CompareRequest,
    db: AsyncSession = Depends(get_db),
    x_llm_provider: Optional[str] = Header(None),
    x_llm_model: Optional[str] = Header(None),
    x_session_id: Optional[str] = Header(None)
):
    if len(body.paper_ids) < 2:
        raise HTTPException(status_code=400, detail='Provide at least 2 paper IDs')

    papers = []
    for pid in body.paper_ids:
        p = await db.get(Paper, pid)
        if not p or p.session_id != x_session_id:
            raise HTTPException(status_code=404, detail=f'Paper {pid} not found')
        papers.append(p)

    # For each paper, retrieve top-k chunks relevant to the comparison aspect
    emb = EmbeddingService()
    vs = VectorStore()
    query_vec = emb.embed(body.aspect)

    paper_contexts = []
    for paper in papers:
        chunks = vs.query(query_vec, n_results=4, paper_id=paper.id)
        context = '\n'.join([c['text'] for c in chunks])
        paper_contexts.append(f'=== {paper.title} ({paper.year or "n.d."}) ===\n{context}')

    combined = '\n\n'.join(paper_contexts)
    user_prompt = f"""Compare these {len(papers)} papers on: {body.aspect}

{combined}

Provide a structured comparison:"""

    import httpx
    llm = LLMService(provider=x_llm_provider, model=x_llm_model)
    try:
        comparison = await llm.complete(COMPARE_SYSTEM, user_prompt, max_tokens=1500)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Failed to connect to the LLM provider '{llm.provider}'. "
                "If using Ollama, make sure the Ollama desktop application is started on your computer. "
                "If using Groq, Gemini, or Qwen, make sure your API key is correctly set in your .env file."
            )
        )
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        try:
            err_json = e.response.json()
            if "error" in err_json and isinstance(err_json["error"], dict) and "message" in err_json["error"]:
                detail = f"LLM Provider Error: {err_json['error']['message']}"
            else:
                detail = f"HTTP {status_code}: {e.response.text}"
        except Exception:
            detail = f"The LLM provider '{llm.provider}' returned an HTTP error: {e.response.text}"
        raise HTTPException(status_code=status_code, detail=detail)

    return {
        'papers': [{'id': p.id, 'title': p.title, 'year': p.year} for p in papers],
        'aspect': body.aspect,
        'comparison': comparison
    }
