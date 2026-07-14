from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    ollama_host: str = 'http://localhost:11434'
    llm_provider: str = 'openrouter'
    llm_model: str = 'qwen/qwen-2.5-72b-instruct'
    chroma_persist_dir: str = './chroma_db'
    upload_dir: str = './uploads'
    database_url: str = 'sqlite+aiosqlite:///./papers.db'
    embedding_model: str = 'all-MiniLM-L6-v2'
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_chunks_per_query: int = 5
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    qwen_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None

    class Config:
        env_file = '.env'

settings = Settings()
