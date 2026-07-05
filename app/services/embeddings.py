import httpx
from app.config import settings
from typing import List

class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            import sys
            if not settings.gemini_api_key or "pytest" in sys.modules:
                # Dynamic import to avoid loading PyTorch into memory on Render
                from sentence_transformers import SentenceTransformer
                cls._model = SentenceTransformer(settings.embedding_model)
        return cls._instance

    def embed(self, text: str) -> List[float]:
        import sys
        if settings.gemini_api_key and "pytest" not in sys.modules:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={settings.gemini_api_key}"
            payload = {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": text}]},
                "outputDimensionality": 384
            }
            with httpx.Client(timeout=30.0) as client:
                r = client.post(url, json=payload)
                r.raise_for_status()
                return r.json()["embedding"]["values"]
        else:
            return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        import sys
        if settings.gemini_api_key and "pytest" not in sys.modules:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key={settings.gemini_api_key}"
            requests_list = [
                {
                    "model": "models/gemini-embedding-001",
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": 384
                } for t in texts
            ]
            all_embeddings = []
            with httpx.Client(timeout=60.0) as client:
                for i in range(0, len(texts), 100):
                    chunk = requests_list[i:i+100]
                    payload = {"requests": chunk}
                    r = client.post(url, json=payload)
                    r.raise_for_status()
                    for emb in r.json().get("embeddings", []):
                        all_embeddings.append(emb["values"])
            return all_embeddings
        else:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100
            )
            return embeddings.tolist()
