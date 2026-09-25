"""
Stage 2: Repo Indexer
----------------------
Walks a directory of Python files, splits each file into function/class-level
chunks using the `ast` module (not blind character-count chunking - that's
the whole point of this stage), and stores them in a local ChromaDB
collection with embeddings computed entirely on your machine (no API key,
no cost, no rate limit).

Why AST-aware chunking instead of "every N characters"?
Character-based chunking is the default in most RAG tutorials because it's
simple, but it can slice a function in half, separating a docstring from
its body or splitting a loop mid-statement. That gives the model a
half-formed piece of context to answer from. Chunking by function/class
guarantees every chunk is a complete, meaningful unit of code.

Usage:
    python ingest.py .                  # index the current directory
    python ingest.py /path/to/other/repo
"""

import argparse
import ast
import json
import os
import sys

import chromadb

DB_PATH = "./chroma_db"
COLLECTION_NAME = "codebase"

# Directories we never want to walk into.
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "chroma_db"}


def find_python_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def chunk_file(file_path: str):
    """Yield one chunk per top-level function/class (and methods inside
    classes), each with its source code and line-number metadata."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        print(f"  [skip] could not parse {file_path} (syntax error)")
        return

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            code = ast.get_source_segment(source, node)
            if not code:
                continue
            yield {
                "code": code,
                "symbol": node.name,
                "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                "start_line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
            }

            # Also index methods inside classes as their own chunks, so a
            # question about one method doesn't force retrieving the whole
            # class.
            if isinstance(node, ast.ClassDef):
                for sub in ast.iter_child_nodes(node):
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        sub_code = ast.get_source_segment(source, sub)
                        if not sub_code:
                            continue
                        yield {
                            "code": sub_code,
                            "symbol": f"{node.name}.{sub.name}",
                            "kind": "method",
                            "start_line": sub.lineno,
                            "end_line": getattr(sub, "end_lineno", sub.lineno),
                        }


def build_index(root: str):
    client = chromadb.PersistentClient(path=DB_PATH)
    # Fresh index every run - simplest correct behavior for a learning
    # project. A production version would diff and update incrementally.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids, documents, metadatas = [], [], []
    all_chunks = []  # sidecar copy for BM25 keyword search (Stage 3)
    file_count = 0

    for file_path in find_python_files(root):
        file_count += 1
        # Always normalize to forward slashes, even on Windows - otherwise
        # chunk ids and metadata differ by OS (os.path.relpath uses "\" on
        # Windows), which silently breaks any comparison against a fixed
        # reference like eval_set.json.
        rel_path = os.path.relpath(file_path, root).replace(os.sep, "/")
        for chunk in chunk_file(file_path):
            chunk_id = f"{rel_path}::{chunk['symbol']}::{chunk['start_line']}"
            ids.append(chunk_id)
            documents.append(chunk["code"])
            metadata = {
                "file": rel_path,
                "symbol": chunk["symbol"],
                "kind": chunk["kind"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
            }
            metadatas.append(metadata)
            all_chunks.append({"id": chunk_id, "code": chunk["code"], **metadata})

    if not documents:
        print("No chunks found. Is the path correct, and does it contain .py files?")
        sys.exit(1)

    # ChromaDB computes embeddings locally using its default embedding
    # model on first use (downloads once, then fully offline and free).
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    # BM25 (keyword search) works on raw text, not vectors, so it can't
    # live inside Chroma - we save the same chunks as plain JSON for
    # ask.py / eval.py to build a keyword index from at query time.
    chunks_path = os.path.join(DB_PATH, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Indexed {len(documents)} chunks from {file_count} file(s) into '{COLLECTION_NAME}'.")
    print(f"Vector store saved at {DB_PATH}/")
    print(f"Chunk sidecar (for keyword search) saved at {chunks_path}")


def main():
    parser = argparse.ArgumentParser(description="Index a Python repo into a local vector store.")
    parser.add_argument("path", nargs="?", default=".", help="Root directory to index (default: current dir).")
    args = parser.parse_args()
    build_index(args.path)


if __name__ == "__main__":
    main()