"""Ingest PDFs: extract text per page, chunk it, store in ChromaDB."""
from pathlib import Path

import chromadb
from pypdf import PdfReader

RAW_DIR = Path("data/raw")
DB_DIR = "data/chroma_db"
COLLECTION = "pdf_docs"


def get_collection(name=COLLECTION):
    """Open (or create) a ChromaDB collection by name."""
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(name)


def chunk_text(text, size=800, overlap=150):
    """Split text into overlapping chunks of about `size` characters."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def ingest_pdf(file, name, collection=None):
    """Ingest one PDF (a path or file-like object). Returns number of chunks."""
    if collection is None:
        collection = get_collection()
    reader = PdfReader(file)
    total = 0
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        chunks = chunk_text(text)
        if not chunks:
            continue
        collection.upsert(
            ids=[f"{name}-p{page_no}-c{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"source": name, "page": page_no}] * len(chunks),
        )
        total += len(chunks)
    return total


def ingest():
    """Ingest every PDF in data/raw."""
    collection = get_collection()
    for pdf in sorted(RAW_DIR.glob("*.pdf")):
        n = ingest_pdf(str(pdf), pdf.name, collection)
        print(f"Ingested {pdf.name}: {n} chunks")
    print("Total chunks:", collection.count())


if __name__ == "__main__":
    ingest()