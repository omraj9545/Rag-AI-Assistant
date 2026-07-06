from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models import Paper
from app.services.ingestion import ingest_paper
from app.config import settings
import shutil
import os
from typing import Optional

router = APIRouter()

@router.post('/upload')
async def upload_paper(
    file: UploadFile = File(...), 
    db: AsyncSession = Depends(get_db),
    x_session_id: Optional[str] = Header(None)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDF files are accepted')
    
    save_path = os.path.join(settings.upload_dir, file.filename)
    # Ensure upload directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        paper = await ingest_paper(save_path, file.filename, db, session_id=x_session_id)
        return {'paper_id': paper.id, 'filename': paper.filename, 'title': paper.title}
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@router.get('/')
async def list_papers(
    db: AsyncSession = Depends(get_db),
    x_session_id: Optional[str] = Header(None)
):
    result = await db.execute(select(Paper).where(Paper.session_id == x_session_id))
    papers = result.scalars().all()
    return [{'id': p.id, 'filename': p.filename, 'title': p.title, 'num_pages': p.num_pages, 'num_chunks': p.num_chunks} for p in papers]

@router.get('/{paper_id}')
async def get_paper(
    paper_id: str, 
    db: AsyncSession = Depends(get_db),
    x_session_id: Optional[str] = Header(None)
):
    paper = await db.get(Paper, paper_id)
    if not paper or paper.session_id != x_session_id:
        raise HTTPException(status_code=404, detail='Paper not found')
    return paper

@router.delete('/{paper_id}')
async def delete_paper(
    paper_id: str, 
    db: AsyncSession = Depends(get_db),
    x_session_id: Optional[str] = Header(None)
):
    paper = await db.get(Paper, paper_id)
    if not paper or paper.session_id != x_session_id:
        raise HTTPException(status_code=404, detail='Paper not found')

    # Delete local PDF file
    if paper.file_path and os.path.exists(paper.file_path):
        try:
            os.remove(paper.file_path)
        except Exception:
            pass

    # Delete from vector store
    from app.services.vector_store import VectorStore
    vs = VectorStore()
    try:
        vs.delete_paper(paper_id)
    except Exception:
        pass

    # Delete from database
    await db.delete(paper)
    await db.commit()
    return {'deleted': paper_id}

@router.get('/{paper_id}/summarize')
async def summarize_paper(
    paper_id: str,
    db: AsyncSession = Depends(get_db),
    x_llm_provider: Optional[str] = Header(None),
    x_llm_model: Optional[str] = Header(None),
    x_session_id: Optional[str] = Header(None)
):
    paper = await db.get(Paper, paper_id)
    if not paper or paper.session_id != x_session_id:
        raise HTTPException(status_code=404, detail='Paper not found')

    from app.services.vector_store import VectorStore
    from app.services.summarizer import SummarizerService
    
    vs = VectorStore()
    results = vs._collection.get(
        where={'paper_id': paper_id},
        include=['documents', 'metadatas']
    )
    
    if not results or not results.get('documents'):
        return {'paper_id': paper_id, 'title': paper.title, 'summary': "No content available to summarize."}
        
    chunks = [{'text': t, 'metadata': m}
              for t, m in zip(results['documents'], results['metadatas'])]
    chunks.sort(key=lambda x: x['metadata']['chunk_index'])

    import httpx
    summarizer = SummarizerService(provider=x_llm_provider, model=x_llm_model)
    try:
        summary = await summarizer.summarize_paper(paper, chunks)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Failed to connect to the LLM provider. "
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
            detail = f"The LLM provider returned an HTTP error: {e.response.text}"
        raise HTTPException(status_code=status_code, detail=detail)
    return {'paper_id': paper_id, 'title': paper.title, 'summary': summary}

