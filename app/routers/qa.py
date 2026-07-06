from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models import Paper
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm import LLMService
from typing import Optional

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5

SYSTEM_PROMPT = """You are an expert research assistant.
Answer the user's question using ONLY the provided context excerpts from a research paper.
If the context does not contain enough information, say so clearly.
Always cite which excerpt (by number) supports your answer.
Be precise, technical, and academic in tone."""

@router.post('/{paper_id}/ask')
async def ask_question(
    paper_id: str,
    body: QuestionRequest,
    db: AsyncSession = Depends(get_db),
    x_llm_provider: Optional[str] = Header(None),
    x_llm_model: Optional[str] = Header(None),
    x_session_id: Optional[str] = Header(None)
):
    paper = await db.get(Paper, paper_id)
    if not paper or paper.session_id != x_session_id:
        raise HTTPException(status_code=404, detail='Paper not found')

    # 1. Embed the question
    emb = EmbeddingService()
    q_embedding = emb.embed(body.question)

    # 2. Retrieve top-k relevant chunks
    vs = VectorStore()
    chunks = vs.query(q_embedding, n_results=body.top_k, paper_id=paper_id)

    if not chunks:
        raise HTTPException(status_code=404, detail='No relevant content found in paper')

    # 3. Build context string with numbered excerpts
    context = ''
    for i, chunk in enumerate(chunks, 1):
        context += f'[Excerpt {i} | Page {chunk["metadata"]["page_num"]} | Score: {chunk["score"]:.2f}]\n'
        context += chunk['text'] + '\n\n'

    # 4. Build augmented prompt
    user_prompt = f"""Paper: {paper.title}

Context excerpts from the paper:
{context}

Question: {body.question}

Answer based strictly on the excerpts above:"""

    # 5. Call LLM
    import httpx
    llm = LLMService(provider=x_llm_provider, model=x_llm_model)
    try:
        answer = await llm.complete(SYSTEM_PROMPT, user_prompt)
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
        'question': body.question,
        'answer': answer,
        'paper_id': paper_id,
        'paper_title': paper.title,
        'sources': [
            {
                'excerpt_num': i + 1,
                'text': c['text'][:200] + '...',
                'page': c['metadata']['page_num'],
                'score': c['score']
            } for i, c in enumerate(chunks)
        ]
    }
