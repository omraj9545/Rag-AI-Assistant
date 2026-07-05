import fitz   # PyMuPDF
import re

def extract_metadata(pdf_path: str) -> dict:
    doc = fitz.open(pdf_path)
    meta = doc.metadata  # has: title, author, creationDate, etc.

    # Fallback: scrape first page text for title heuristic
    first_page_text = doc[0].get_text('text') if doc.page_count > 0 else ''
    lines = [l.strip() for l in first_page_text.split('\n') if l.strip()]

    title = meta.get('title') or (lines[0] if lines else 'Unknown')
    authors = meta.get('author') or ''

    # Extract year from text using regex
    year_match = re.search(r'\b(19|20)\d{2}\b', first_page_text)
    year = int(year_match.group()) if year_match else None

    # Extract abstract (look for 'Abstract' keyword in first 2 pages)
    abstract = ''
    for i in range(min(2, doc.page_count)):
        text = doc[i].get_text('text')
        abs_match = re.search(r'[Aa]bstract[.:\n](.{100,1500}?)(?:Introduction|1\.)', text, re.DOTALL)
        if abs_match:
            abstract = abs_match.group(1).strip()
            break

    num_pages = doc.page_count
    doc.close()
    return {
        'title': title,
        'authors': authors,
        'year': year,
        'abstract': abstract,
        'num_pages': num_pages
    }
