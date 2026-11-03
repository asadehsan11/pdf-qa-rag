"""Summarize one indexed PDF into section cards (heading + 2-line summary)."""
import json
import re

from src.ingest import get_collection
from src.llm import generate

OVERLAP = 150  # must match the chunk overlap used in src/ingest.py
MAX_CHARS = 60000


def get_document_text(name):
    """Rebuild the text of one indexed PDF, with [Page N] markers."""
    data = get_collection().get(
        where={"source": name}, include=["documents", "metadatas"]
    )
    pages = {}
    for chunk_id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        index = int(chunk_id.rsplit("-c", 1)[1])
        pages.setdefault(meta["page"], []).append((index, doc))
    if not pages:
        raise ValueError("No text found for this document.")

    page_texts = {}
    for page, chunks in pages.items():
        chunks.sort()
        page_texts[page] = chunks[0][1] + "".join(doc[OVERLAP:] for _, doc in chunks[1:])

    limit = max(500, MAX_CHARS // len(page_texts))
    return "\n\n".join(
        f"[Page {p}]\n{page_texts[p][:limit]}" for p in sorted(page_texts)
    )


def _parse_cards(raw):
    """Pull a JSON list of cards out of the model's reply."""
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    cards = [
        {
            "heading": str(c["heading"]),
            "summary": str(c["summary"]),
            "pages": str(c.get("pages", "")),
        }
        for c in data
        if isinstance(c, dict) and "heading" in c and "summary" in c
    ]
    return cards or None


def summarize_document(name):
    """Return a list of {heading, summary, pages} cards for one PDF."""
    text = get_document_text(name)
    prompt = (
        "Split the document below into 6 to 10 logical sections. For each section "
        "give: a short heading (max 8 words), a summary of exactly 2 short lines "
        "(max 30 words in total), and the page range as text such as '3-5'. "
        "Use only information from the document. Return ONLY a JSON array, with no "
        "extra text, in this form: "
        '[{"heading": "...", "summary": "...", "pages": "3-5"}]\n\n'
        f"Document:\n{text}"
    )
    for _ in range(2):
        cards = _parse_cards(generate(prompt))
        if cards:
            return cards
    raise ValueError("Could not read the summary. Please try again.")