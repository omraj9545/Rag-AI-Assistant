from fastapi import APIRouter, Query, Depends, Header
from app.database import get_db
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import Paper
from typing import Optional

router = APIRouter()

@router.get('/')
async def semantic_search(
    q: str = Query(..., description='Natural language search query'),
    top_k: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    x_session_id: Optional[str] = Header(None)
):
    print("Semantic Search query received:", q)
    emb = EmbeddingService()
    query_vec = emb.embed(q)

    vs = VectorStore()
    results = vs.query(query_vec, n_results=top_k, session_id=x_session_id)

    # Enrich search results with SQLite paper metadata
    enriched = []
    for r in results:
        pid = r['metadata']['paper_id']
        paper = await db.get(Paper, pid)
        enriched.append({
            'score': r['score'],
            'text': r['text'],
            'page': r['metadata']['page_num'],
            'paper': {
                'id': pid,
                'title': paper.title if paper else 'Unknown',
                'year': paper.year if paper else None
            }
        })

    return {'query': q, 'results': enriched}
