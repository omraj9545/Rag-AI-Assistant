from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.database import init_db
from app.routers import papers, qa, search, compare
from app.config import settings
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure folders exist
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    # Initialize SQLite database schema
    await init_db()
    yield

app = FastAPI(
    title='AI Research Assistant',
    description='RAG-powered research paper Q&A system',
    version='1.0.0',
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Register routers
app.include_router(papers.router,  prefix='/papers',  tags=['Papers'])
app.include_router(qa.router,      prefix='/papers',  tags=['Q&A'])
app.include_router(search.router,  prefix='/search',  tags=['Search'])
app.include_router(compare.router, prefix='/compare', tags=['Compare'])

@app.get('/')
async def root():
    return {'status': 'ok', 'version': '1.0.0'}
