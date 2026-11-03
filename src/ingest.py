"""Ingest PDFs: extract text per page, chunk it, store in ChromaDB."""
from pathlib import Path

import chromadb
from pypdf import PdfReader

RAW_DIR = Path("data/raw")
DB_DIR = "data/chroma_db"
COLLECTION = "pdf_docs"


def chunk_text(text, size=800, overlap=150):
    """Split text into overlapping chunks of about `size` characters."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def ingest():
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(COLLECTION)
    for pdf in sorted(RAW_DIR.glob("*.pdf")):
        reader = PdfReader(str(pdf))
        for page_no, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            chunks = chunk_text(text)
            if not chunks:
                continue
            collection.upsert(
                ids=[f"{pdf.name}-p{page_no}-c{i}" for i in range(len(chunks))],
                documents=chunks,
                metadatas=[{"source": pdf.name, "page": page_no}] * len(chunks),
            )
        print(f"Ingested {pdf.name} ({len(reader.pages)} pages)")
    print("Total chunks:", collection.count())


if __name__ == "__main__":
    ingest()