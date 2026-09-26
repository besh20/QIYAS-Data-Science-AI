"""Evaluates RAG retrieval: for each test query, checks whether the correct
knowledge-base section is retrieved in the top-k results. This needs the
embedding model (sentence-transformers), so it requires internet the first
time it downloads the model, then works offline from cache.

Run: python tests/evaluate_retrieval.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag import RagIndex

# (query, keyword that should appear in the top-1 retrieved chunk's header)
# Update this list to match your REAL knowledge_base.md section headers
# once you swap in real content.
TEST_CASES = [
    ("how much is the registration fee", "Registration Fee"),
    ("ስንት ብር ነው ክፍያው?", "Registration Fee"),
    ("what documents do I need", "Registration Requirements"),
    ("ምን አይነት ወረቀት ያስፈልጋል?", "Registration Requirements"),
    ("when does registration open", "Registration Dates"),
    ("registration meche yikefetal?", "Registration Dates"),
    ("where is the office located", "Where to Register"),
    ("registration office yet new?", "Where to Register"),
    ("my ID card is lost", "Common Problems"),
    ("I have a complaint who do I talk to", "Contact"),
]


def main():
    index = RagIndex()
    correct_top1 = 0
    correct_top3 = 0
    rows = []

    for query, expected_keyword in TEST_CASES:
        results = index.retrieve(query, k=3)
        top1_hit = expected_keyword.lower() in results[0][0].lower()
        top3_hit = any(expected_keyword.lower() in chunk.lower() for chunk, _ in results)
        correct_top1 += int(top1_hit)
        correct_top3 += int(top3_hit)
        rows.append((query, expected_keyword, top1_hit, top3_hit, results[0][1]))
        mark = "OK" if top1_hit else ("~top3" if top3_hit else "MISS")
        print(f"[{mark}] '{query}' -> expected '{expected_keyword}' (top1 score={results[0][1]:.3f})")

    n = len(TEST_CASES)
    top1_acc = correct_top1 / n
    top3_acc = correct_top3 / n
    print(f"\nTop-1 retrieval accuracy: {correct_top1}/{n} = {top1_acc:.0%}")
    print(f"Top-3 retrieval accuracy: {correct_top3}/{n} = {top3_acc:.0%}")

    report_path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_report_retrieval.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# RAG Retrieval -- Evaluation\n\n")
        f.write(f"- Top-1 accuracy: **{correct_top1}/{n} = {top1_acc:.0%}**\n")
        f.write(f"- Top-3 accuracy: **{correct_top3}/{n} = {top3_acc:.0%}**\n\n")
        f.write("| Query | Expected section | Top-1 hit | Top-3 hit | Top score |\n|---|---|---|---|---|\n")
        for query, expected, t1, t3, score in rows:
            f.write(f"| {query} | {expected} | {'yes' if t1 else 'no'} | {'yes' if t3 else 'no'} | {score:.3f} |\n")

    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
