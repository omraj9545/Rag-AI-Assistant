import chromadb
from app.config import settings
from typing import List, Dict, Optional

class VectorStore:
    _client = None
    _collection = None

    def __init__(self):
        if VectorStore._client is None:
            VectorStore._client = chromadb.PersistentClient(
                path=settings.chroma_persist_dir
            )
            VectorStore._collection = VectorStore._client.get_or_create_collection(
                name='paper_chunks',
                metadata={'hnsw:space': 'cosine'}  # cosine similarity
            )

    def add_chunks(self, paper_id: str, chunk_ids: List[str],
                   texts: List[str], embeddings: List[List[float]],
                   metadatas: List[Dict]):
        self._collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def query(self, query_embedding: List[float], n_results: int = 5,
              paper_id: Optional[str] = None) -> List[Dict]:
        where = {'paper_id': paper_id} if paper_id else None
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results or not results.get('ids') or len(results['ids']) == 0:
            return []
            
        ids = results['ids'][0]
        if not ids:
            return []
            
        chunks = []
        for i in range(len(ids)):
            chunks.append({
                'id': ids[i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i]  # cosine similarity
            })
        return chunks

    def delete_paper(self, paper_id: str):
        results = self._collection.get(where={'paper_id': paper_id})
        if results['ids']:
            self._collection.delete(ids=results['ids'])
