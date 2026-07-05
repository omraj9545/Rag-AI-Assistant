import re
from typing import List, Tuple

def split_into_sentences(text: str) -> List[str]:
    # Split on sentence-ending punctuation followed by whitespace/newline
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Returns list of chunk_text strings.
    Strategy: fill chunks by word count, respect sentence boundaries,
    add overlap from previous chunk for context continuity.
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) > chunk_size and current_chunk:
            chunk_text_str = ' '.join(current_chunk)
            chunks.append(chunk_text_str)
            # Keep last `overlap` words as prefix for next chunk
            all_words = chunk_text_str.split()
            overlap_words = all_words[-overlap:] if len(all_words) > overlap else all_words
            current_chunk = overlap_words + words
            current_len = len(current_chunk)
        else:
            current_chunk.extend(words)
            current_len += len(words)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def extract_text_by_page(pdf_path: str) -> List[Tuple[str, int]]:
    import fitz
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text('text')
        text = ' '.join(text.split())  # normalize whitespace
        pages.append((text, i + 1))   # (text, page_number)
    doc.close()
    return pages
