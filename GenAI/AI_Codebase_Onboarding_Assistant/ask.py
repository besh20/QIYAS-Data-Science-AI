"""
Stage 2: Ask questions about the indexed repo
------------------------------------------------
Embeds the user's question, retrieves the most relevant code chunks from
the local ChromaDB index built by ingest.py, and asks Groq to answer using
ONLY those chunks - with citations back to exact file/symbol/line.

This is the core RAG loop:
  question -> embed -> vector search -> retrieved chunks -> prompt -> answer

Why force citations in the schema (same structured-output pattern as
Stage 1)?
An answer with no citation is unverifiable - you have no way to tell if
the model is grounded in your actual code or just guessing plausibly.
Forcing every answer to name the file/symbol it drew from makes retrieval
quality checkable, which matters a lot once Stage 3 adds an eval set.

Usage:
    python ask.py "where is risk assessed in this codebase?"
    python ask.py "how does the code chunker handle syntax errors?" --k 5
"""

import argparse
import json
import os
import sys

import chromadb
from dotenv import load_dotenv
from groq import Groq

from retrieval import build_bm25_index, vector_search, keyword_search, hybrid_search

load_dotenv()

DB_PATH = "./chroma_db"
COLLECTION_NAME = "codebase"
MODEL = "openai/gpt-oss-120b"

ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "report_answer",
        "description": "Report a grounded answer to a question about a codebase.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "A clear answer to the question, based ONLY on the "
                                    "provided code chunks. If the chunks don't contain "
                                    "enough information, say so explicitly rather than guessing."
                },
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string"},
                            "symbol": {"type": "string"},
                            "why_relevant": {
                                "type": "string",
                                "description": "One short phrase on why this chunk supports the answer."
                            }
                        },
                        "required": ["file", "symbol", "why_relevant"]
                    },
                    "description": "Every file/symbol actually used to construct the answer."
                },
                "grounded": {
                    "type": "boolean",
                    "description": "True if the answer is fully supported by the retrieved "
                                    "chunks. False if the model had to guess or generalize "
                                    "beyond what was retrieved."
                }
            },
            "required": ["answer", "citations", "grounded"]
        }
    }
}

SYSTEM_PROMPT = """You are a senior engineer helping a new hire understand a codebase. \
You answer questions using ONLY the code chunks provided to you - never invent behavior \
that isn't shown in the chunks. If the retrieved chunks don't fully answer the question, \
say so plainly and set grounded to false rather than filling gaps with assumptions. \
Always call the report_answer tool - never respond in plain text."""


def retrieve(collection, bm25, chunks, question: str, k: int, mode: str):
    if mode == "vector":
        return vector_search(collection, question, k)
    if mode == "keyword":
        return keyword_search(bm25, chunks, question, k)
    return hybrid_search(collection, bm25, chunks, question, k)


def format_context(chunks):
    parts = []
    for c in chunks:
        header = f"# {c['file']} :: {c['symbol']} (lines {c['start_line']}-{c['end_line']})"
        parts.append(f"{header}\n{c['code']}")
    return "\n\n".join(parts)


def answer_question(client, question: str, chunks):
    context = format_context(chunks)
    user_message = (
        f"Retrieved code chunks:\n\n{context}\n\n"
        f"Question: {question}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1500,
        tools=[ANSWER_TOOL],
        tool_choice={"type": "function", "function": {"name": "report_answer"}},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    message = response.choices[0].message
    if message.tool_calls:
        call = message.tool_calls[0]
        if call.function.name == "report_answer":
            return json.loads(call.function.arguments)

    raise RuntimeError("Model did not return the expected tool call.")


def print_report(question: str, retrieved, report: dict):
    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("=" * 60)
    print("\nRETRIEVED CHUNKS (what the model was allowed to see):")
    for c in retrieved:
        print(f"  - {c['file']} :: {c['symbol']} (lines {c['start_line']}-{c['end_line']})")

    print(f"\nGROUNDED: {report['grounded']}")
    print("\nANSWER:")
    print(report["answer"])
    print("\nCITATIONS:")
    for c in report["citations"]:
        print(f"  - {c['file']} :: {c['symbol']} — {c['why_relevant']}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Ask a question about the indexed codebase.")
    parser.add_argument("question", help="Your question, in quotes.")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve (default: 4).")
    parser.add_argument("--mode", choices=["vector", "keyword", "hybrid"], default="hybrid",
                         help="Retrieval strategy (default: hybrid).")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of formatted output.")
    args = parser.parse_args()

    if not os.path.isdir(DB_PATH):
        print(f"No index found at {DB_PATH}/. Run 'python ingest.py .' first.")
        sys.exit(1)

    chunks_path = os.path.join(DB_PATH, "chunks.json")
    if not os.path.isfile(chunks_path):
        print(f"No chunk sidecar found at {chunks_path}. Re-run 'python ingest.py .' "
              f"(this file is needed for keyword/hybrid search).")
        sys.exit(1)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Set GROQ_API_KEY in your .env file (see .env.example).")
        sys.exit(1)

    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    groq_client = Groq(api_key=api_key)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    bm25 = build_bm25_index(chunks)

    retrieved = retrieve(collection, bm25, chunks, args.question, args.k, args.mode)
    if not retrieved:
        print("Nothing retrieved from the index - is it empty? Try re-running ingest.py.")
        sys.exit(1)

    report = answer_question(groq_client, args.question, retrieved)

    if args.json:
        print(json.dumps({"retrieved": retrieved, "report": report}, indent=2))
    else:
        print_report(args.question, retrieved, report)


if __name__ == "__main__":
    main()
