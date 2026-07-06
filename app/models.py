from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship, DeclarativeBase
from datetime import datetime
import uuid

class Base(DeclarativeBase):
    pass

class Paper(Base):
    __tablename__ = 'papers'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=True)
    authors = Column(String, nullable=True)  # Stored as string or JSON-serialized string
    year = Column(Integer, nullable=True)
    abstract = Column(Text, nullable=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    num_pages = Column(Integer, default=0)
    num_chunks = Column(Integer, default=0)
    session_id = Column(String, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    chunks = relationship('Chunk', back_populates='paper', cascade='all, delete')

class Chunk(Base):
    __tablename__ = 'chunks'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    paper_id = Column(String, ForeignKey('papers.id'), nullable=False)
    text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_num = Column(Integer, nullable=True)
    
    paper = relationship('Paper', back_populates='chunks')
