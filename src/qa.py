"""Answer questions from the ingested PDFs, with citations."""
import time

from src.db import log_qa
from src.ingest import get_collection
from src.llm import generate


def retrieve(question, k=5):
    """Find the k chunks most similar to the question."""
    result = get_collection().query(query_texts=[question], n_results=k)
    return [
        {"text": doc, "source": meta["source"], "page": meta["page"]}
        for doc, meta in zip(result["documents"][0], result["metadatas"][0])
    ]


def answer(question, k=5):
    """Answer a question using only the retrieved chunks, and log it."""
    start = time.time()
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
    text = generate(prompt)
    log_qa(
        question,
        text,
        [{"source": c["source"], "page": c["page"]} for c in chunks],
        round(time.time() - start, 2),
    )
    return {"answer": text, "sources": chunks}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is this document about?"
    result = answer(q)
    print(result["answer"])
    print("\nSources:")
    for i, c in enumerate(result["sources"], start=1):
        print(f"[{i}] {c['source']} - page {c['page']}")