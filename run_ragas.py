"""Score the RAG pipeline with RAGAS and save the results."""
import json
import os
import time
import warnings
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from google import genai
from ragas import evaluate
from ragas.embeddings import GoogleEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import AnswerRelevancy, Faithfulness
from ragas.run_config import RunConfig

from src.qa import answer

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

JUDGE_MODEL = "gemini-flash-lite-latest"
PAUSE_SECONDS = 20  # keeps us under the free-tier limit of 15 requests per minute
CACHE = Path("reports/ragas_answers.json")
QUESTIONS = [
    "What is Wireshark used for?",
    "What is the difference between capture filters and display filters in Wireshark?",
    "What is virtualization?",
    "What is the difference between Type-1 and Type-2 hypervisors?",
    "What is a snapshot used for in a virtual machine?",
    "Define histogram.",
    "How are images represented as arrays of pixel values in NumPy?",
    "What is the Normal Equation used for in the regression lab?",
]


class GoogleEmbeddingsCompat(GoogleEmbeddings):
    """Adds the method names that the classic AnswerRelevancy metric calls."""

    def embed_query(self, text):
        return self.embed_text(text)

    def embed_documents(self, texts):
        return [self.embed_text(t) for t in texts]


def collect_answers():
    """Ask the app every question once and cache the answers and contexts."""
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    rows = []
    for q in QUESTIONS:
        result = answer(q)
        rows.append(
            {
                "question": q,
                "answer": result["answer"],
                "contexts": [c["text"] for c in result["sources"]],
            }
        )
        print("Answered:", q)
        time.sleep(2)
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def score_row(row, metrics):
    """Score one question, retrying if a temporary error left a score empty."""
    data = Dataset.from_dict(
        {
            "question": [row["question"]],
            "answer": [row["answer"]],
            "contexts": [row["contexts"]],
        }
    )
    f = r = None
    for attempt in range(3):
        result = evaluate(
            data,
            metrics=metrics,
            run_config=RunConfig(max_workers=1, timeout=180),
            show_progress=False,
        )
        scores = result.to_pandas().iloc[0]
        f = scores.get("faithfulness")
        r = scores.get("answer_relevancy")
        if pd.notna(f) and pd.notna(r):
            break
        print(f"  attempt {attempt + 1}: a score is missing, waiting and retrying...")
        time.sleep(PAUSE_SECONDS)
    return (None if pd.isna(f) else float(f)), (None if pd.isna(r) else float(r))


def main():
    rows = collect_answers()

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    llm = llm_factory(JUDGE_MODEL, provider="google", client=client)
    embeddings = GoogleEmbeddingsCompat(client=client, model="gemini-embedding-001")
    metrics = [Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=embeddings)]

    results = []
    for i, row in enumerate(rows, start=1):
        print(f"Scoring {i}/{len(rows)}: {row['question']}")
        f, r = score_row(row, metrics)
        results.append(
            {"question": row["question"], "faithfulness": f, "answer_relevancy": r}
        )
        print(f"  faithfulness={f}  answer_relevancy={r}")
        time.sleep(PAUSE_SECONDS)

    df = pd.DataFrame(results)
    df.to_csv("reports/ragas_results.csv", index=False)
    print("\nAverage scores:")
    print(df[["faithfulness", "answer_relevancy"]].mean())


if __name__ == "__main__":
    main()