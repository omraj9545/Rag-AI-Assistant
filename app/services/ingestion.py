from app.utils.chunker import extract_text_by_page, chunk_text
from app.utils.metadata import extract_metadata
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.config import settings
from app.models import Paper, Chunk
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

async def ingest_paper(file_path: str, filename: str, db: AsyncSession, session_id: str = None) -> Paper:
    # 1. Extract metadata
    meta = extract_metadata(file_path)

    # 2. Create Paper record in SQLite
    paper = Paper(
        id=str(uuid.uuid4()),
        title=meta['title'],
        authors=meta['authors'],
        year=meta['year'],
        abstract=meta['abstract'],
        filename=filename,
        file_path=file_path,
        num_pages=meta['num_pages'],
        session_id=session_id
    )
    db.add(paper)
    await db.flush()  # Get paper.id without doing a full commit

    # 3. Extract text per page & chunk
    pages = extract_text_by_page(file_path)
    all_chunks = []
    chunk_index = 0
    for page_text, page_num in pages:
        page_chunks = chunk_text(page_text, settings.chunk_size, settings.chunk_overlap)
        for c in page_chunks:
            chunk = Chunk(
                id=str(uuid.uuid4()),
                paper_id=paper.id,
                text=c,
                chunk_index=chunk_index,
                page_num=page_num
            )
            db.add(chunk)
            all_chunks.append(chunk)
            chunk_index += 1

    paper.num_chunks = chunk_index

    if all_chunks:
        # 4. Embed all chunks
        emb_service = EmbeddingService()
        texts = [c.text for c in all_chunks]
        embeddings = emb_service.embed_batch(texts)

        # 5. Store in ChromaDB
        vs = VectorStore()
        vs.add_chunks(
            paper_id=paper.id,
            chunk_ids=[c.id for c in all_chunks],
            texts=texts,
            embeddings=embeddings,
            metadatas=[{
                'paper_id': paper.id,
                'page_num': c.page_num,
                'chunk_index': c.chunk_index,
                'session_id': session_id or ''
            } for c in all_chunks]
        )

    await db.commit()
    await db.refresh(paper)
    return paper
