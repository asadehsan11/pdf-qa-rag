"""Answer questions from the ingested PDFs, with citations."""
import chromadb

from src.llm import generate

DB_DIR = "data/chroma_db"
COLLECTION = "pdf_docs"

_collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)


def retrieve(question, k=5):
    """Find the k chunks most similar to the question."""
    result = _collection.query(query_texts=[question], n_results=k)
    return [
        {"text": doc, "source": meta["source"], "page": meta["page"]}
        for doc, meta in zip(result["documents"][0], result["metadatas"][0])
    ]


def answer(question, k=5):
    """Answer a question using only the retrieved chunks."""
    chunks = retrieve(question, k)
    context = "\n\n".join(
        f"[{i}] ({c['source']}, page {c['page']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "Cite sources like [1], [2]. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return {"answer": generate(prompt), "sources": chunks}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is this document about?"
    result = answer(q)
    print(result["answer"])
    print("\nSources:")
    for i, c in enumerate(result["sources"], start=1):
        print(f"[{i}] {c['source']} - page {c['page']}")