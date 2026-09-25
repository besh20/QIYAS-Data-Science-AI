"""
Stage 4: Tool-calling agent (ReAct-style: Reason -> Act -> Observe, repeat)
------------------------------------------------------------------------------
Give it a bug description in plain English. Unlike Stages 1-3, this does
NOT retrieve once and answer - it runs a LOOP where the model itself
decides, turn by turn, whether to search the codebase again (with a
refined query) or whether it has enough to propose a fix.

This is the actual mechanical definition of "agent" used here: a model
that can call tools, see the results, and decide its own next step -
not a fixed pipeline you wrote in advance.

Design decision: propose-only, no file edits (yet).
This stage proves out the REASONING loop - search, evaluate, refine,
decide - which is the hard part conceptually. Actually writing the fix
to disk is a small, separate addition once this loop is trustworthy;
bolting it on now would add risk without adding to what this stage is
meant to teach.

Design decision: the search tool IS Stage 3's hybrid_search.
The agent's search quality is only as good as the retrieval work already
proven out (and measured!) in eval.py. That continuity - Stage 3's eval
results literally becoming Stage 4's tool - is the throughline of this
whole project.

Usage:
    python agent.py "Users report that a malicious user_id crashes the app or leaks data"
    python agent.py "add_item seems to remember items between unrelated calls" --max-steps 5
"""

import argparse
import json
import os
import sys
import time

import chromadb
from dotenv import load_dotenv
from groq import Groq
import groq

from retrieval import build_bm25_index, hybrid_search

load_dotenv()

DB_PATH = "./chroma_db"
COLLECTION_NAME = "codebase"
MODEL = "openai/gpt-oss-120b"
DEFAULT_MAX_STEPS = 6
FORCE_PROPOSE_AFTER_SEARCHES = 2  # after this many searches, stop offering
                                   # search as an option - force a decision

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_codebase",
        "description": "Search the indexed codebase for relevant functions/classes using "
                        "hybrid (semantic + keyword) search. Results may show truncated code - "
                        "that's enough to judge relevance. Call this as many times as needed, "
                        "refining your query, before proposing a fix.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language or keyword search query."},
                "k": {"type": "integer", "description": "How many results to retrieve (default 3, max 4)."}
            },
            "required": ["query"]
        }
    }
}

PROPOSE_FIX_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_fix",
        "description": "Propose a fix once you have located the actual buggy code via "
                        "search_codebase. This ends the investigation - only call this "
                        "once you are confident, and only after at least one search.",
        "parameters": {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "symbol": {"type": "string"},
                "explanation": {
                    "type": "string",
                    "description": "What's wrong and why this fix addresses it."
                },
                "original_code": {
                    "type": "string",
                    "description": "The exact original code, copied from what search_codebase returned."
                },
                "fixed_code": {
                    "type": "string",
                    "description": "The corrected version of the code."
                },
                "confidence": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "How confident you are this fix is correct and complete."
                }
            },
            "required": ["file", "symbol", "explanation", "original_code", "fixed_code", "confidence"]
        }
    }
}

SYSTEM_PROMPT = """You are an autonomous code-fixing agent helping with an unfamiliar codebase. \
You will be given a bug description in plain English. You do NOT know the codebase in advance - \
you must use the search_codebase tool to locate the actual relevant code before proposing anything. \

Formulate search queries about SYMPTOMS, BEHAVIOR, or CONCEPTS (e.g. "user id used directly in a \
database query", "global mutable cache", "function with a list default argument") - NOT literal \
filenames or paths. The search index matches code semantics and keywords, not file paths, so \
searching for a filename like "example_functions.py" will not reliably find the right function.

Do not repeat a search you have already run - if you already retrieved a function's code in an \
earlier search, you already have it available above in this conversation; re-running the same \
query wastes a step and will return the same result. As soon as you have found and can see the \
actual buggy code, stop searching and call propose_fix. You do not need to search more than 1-2 \
times for a focused bug report.

Never propose a fix for code you have not actually seen via search_codebase - do not guess at code \
you haven't retrieved. When you are confident you've found the right code and have a correct fix, \
call propose_fix exactly once to end the investigation."""


def setup():
    if not os.path.isdir(DB_PATH):
        print(f"No index found at {DB_PATH}/. Run 'python ingest.py .' first.")
        sys.exit(1)
    chunks_path = os.path.join(DB_PATH, "chunks.json")
    if not os.path.isfile(chunks_path):
        print(f"No chunk sidecar found. Re-run 'python ingest.py .'.")
        sys.exit(1)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Set GROQ_API_KEY in your .env file.")
        sys.exit(1)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    bm25 = build_bm25_index(chunks)
    groq_client = Groq(api_key=api_key)

    return groq_client, collection, bm25, chunks


def run_search_tool(collection, bm25, chunks, args: dict):
    query = args["query"]
    # Cap k regardless of what the model requests - keeps every single
    # tool result small, which matters a lot on a free-tier token budget.
    k = min(args.get("k", 3), 4)
    results = hybrid_search(collection, bm25, chunks, query, k)

    MAX_CODE_CHARS = 500  # enough to judge relevance, not a full essay of code
    trimmed = []
    for r in results:
        code = r["code"]
        if len(code) > MAX_CODE_CHARS:
            code = code[:MAX_CODE_CHARS] + "\n... (truncated - this is enough to confirm relevance; " \
                   "if you need the FULL code to write a fix, search again with a query naming " \
                   "this exact symbol, e.g. '" + r["symbol"] + " full implementation')"
        trimmed.append({
            "file": r["file"],
            "symbol": r["symbol"],
            "start_line": r["start_line"],
            "end_line": r["end_line"],
            "code": code,
        })
    return trimmed


def compress_older_tool_results(messages: list):
    """Keep only the MOST RECENT tool result's full content. Older tool
    results get collapsed to a one-line summary - the agent already
    'saw' that code on an earlier turn, so resending it in full every
    subsequent call just burns tokens for no new information. This is
    the main thing that keeps a multi-turn agent loop under a free-tier
    TPM budget as the conversation grows."""
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= 1:
        return
    for i in tool_indices[:-1]:  # all but the most recent
        try:
            parsed = json.loads(messages[i]["content"])
            if isinstance(parsed, list):
                summary = [f"{c.get('file')}::{c.get('symbol')}" for c in parsed if isinstance(c, dict)]
                messages[i]["content"] = json.dumps({"previously_found_symbols": summary})
        except (json.JSONDecodeError, TypeError):
            pass  # not a search result (e.g. an error message) - leave as-is


def finalize_fix(client, bug_description: str, gathered_chunks: dict):
    """Fallback used when the model fails to comply with a forced
    tool_choice inside the main loop (a real limitation of smaller
    models - forcing tool_choice restricts what the API will ACCEPT,
    but doesn't guarantee the model generates a valid call for it).

    Rather than retry the same long, noisy conversation, this builds a
    short, clean context - just the bug report and the actual code
    already found - which gives the model a much better chance of
    complying with the forced propose_fix call.
    """
    context_parts = []
    for chunk in gathered_chunks.values():
        context_parts.append(
            f"# {chunk['file']} :: {chunk['symbol']} (lines {chunk['start_line']}-{chunk['end_line']})\n"
            f"{chunk['code']}"
        )
    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": "You are finalizing a bug fix. You have already investigated and found "
                       "the relevant code below. Call propose_fix now, using the exact code shown "
                       "as original_code."
        },
        {
            "role": "user",
            "content": f"Bug report: {bug_description}\n\nCode already found:\n\n{context}"
        },
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=800,
            tools=[PROPOSE_FIX_TOOL],
            tool_choice={"type": "function", "function": {"name": "propose_fix"}},
            messages=messages,
        )
    except groq.BadRequestError:
        return None

    message = response.choices[0].message
    if message.tool_calls:
        call = message.tool_calls[0]
        if call.function.name == "propose_fix":
            return json.loads(call.function.arguments)
    return None


def run_agent(client, collection, bm25, chunks, bug_description: str, max_steps: int):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Bug report: {bug_description}"},
    ]
    transcript = []
    search_count = 0
    seen_queries = set()
    gathered_chunks = {}  # id -> chunk, accumulated across all searches this run

    for step in range(max_steps):
        compress_older_tool_results(messages)

        # Deterministic loop-breaking: prompting a weaker model to "stop
        # searching and commit" is unreliable - it can keep preferring the
        # simpler search tool indefinitely even with plenty of information.
        # Forcing tool_choice to propose_fix after N searches guarantees
        # termination without depending on the model's own judgment.
        if search_count >= FORCE_PROPOSE_AFTER_SEARCHES:
            tool_choice = {"type": "function", "function": {"name": "propose_fix"}}
        else:
            tool_choice = "auto"

        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=800,
                tools=[SEARCH_TOOL, PROPOSE_FIX_TOOL],
                tool_choice=tool_choice,
                messages=messages,
            )
        except groq.BadRequestError:
            # A forced tool_choice restricts what the API will ACCEPT, but a
            # smaller model can still attempt to generate a call to the
            # tool it wasn't allowed to use, which the API then rejects.
            # Rather than crash, fall back to a short, clean finalize call
            # using only the code we've already gathered. NOTE: this must
            # be caught BEFORE the broader APIStatusError below, since
            # BadRequestError is a subclass of it - Python matches the
            # first except clause that fits, so order here is load-bearing.
            print(f"\n[step {step}] Model didn't comply with the forced tool choice. "
                  f"Retrying with a clean, minimal context...")
            fix = finalize_fix(client, bug_description, gathered_chunks)
            if fix:
                transcript.append({"step": step, "type": "proposed_fix", "fix": fix})
                return fix, transcript
            print("Finalize fallback also failed. Try increasing --max-steps, or "
                  "rephrasing the bug description to be more specific.")
            return None, transcript
        except groq.APIStatusError as e:
            if e.status_code in (429, 413):
                print(f"\n[step {step}] Hit Groq's free-tier rate limit "
                      f"({e.status_code}). Waiting 20s and retrying once...")
                time.sleep(20)
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        max_tokens=800,
                        tools=[SEARCH_TOOL, PROPOSE_FIX_TOOL],
                        tool_choice=tool_choice,
                        messages=messages,
                    )
                except groq.APIStatusError:
                    print("Still rate-limited after waiting. Try again in a minute, "
                          "or run with a lower --max-steps to send less context.")
                    return None, transcript
            else:
                raise
        message = response.choices[0].message

        if not message.tool_calls:
            # Model responded in plain text instead of calling a tool -
            # nudge it back on track rather than silently failing.
            transcript.append({"step": step, "type": "text_without_tool", "content": message.content})
            messages.append({"role": "assistant", "content": message.content})
            messages.append({
                "role": "user",
                "content": "Please use search_codebase or propose_fix - do not respond in plain text."
            })
            continue

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [tc.model_dump() for tc in message.tool_calls],
        })

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            if fn_name == "search_codebase":
                search_count += 1
                normalized_query = fn_args.get("query", "").strip().lower()
                results = run_search_tool(collection, bm25, chunks, fn_args)
                transcript.append({"step": step, "type": "search", "query": fn_args.get("query"), "results": results})

                for r in results:
                    gathered_chunks[f"{r['file']}::{r['symbol']}"] = r

                if normalized_query in seen_queries:
                    # Loop-breaking nudge: smaller/free models sometimes re-run an
                    # identical query instead of recognizing they already have
                    # enough to act. Rather than silently returning the same
                    # data again (which invites another repeat), tell it plainly.
                    payload = {
                        "note": "You already ran this exact search earlier in this conversation "
                                "and already have these results above. If you have enough "
                                "information, call propose_fix now. Otherwise try a genuinely "
                                "different query - do not repeat this one again.",
                        "results": results,
                    }
                else:
                    payload = results
                seen_queries.add(normalized_query)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(payload),
                })

            elif fn_name == "propose_fix":
                if search_count == 0:
                    # Guard rail: refuse a fix proposed without ever searching.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({"error": "You must call search_codebase at least once before proposing a fix."}),
                    })
                    transcript.append({"step": step, "type": "rejected_premature_fix"})
                    continue
                transcript.append({"step": step, "type": "proposed_fix", "fix": fn_args})
                return fn_args, transcript

    return None, transcript


def print_transcript(transcript):
    print("\n" + "=" * 60)
    print("AGENT TRANSCRIPT")
    print("=" * 60)
    for entry in transcript:
        if entry["type"] == "search":
            print(f"\n[step {entry['step']}] SEARCH: \"{entry['query']}\"")
            for r in entry["results"]:
                print(f"    found: {r['file']} :: {r['symbol']} (lines {r['start_line']}-{r['end_line']})")
        elif entry["type"] == "rejected_premature_fix":
            print(f"\n[step {entry['step']}] BLOCKED: tried to propose a fix without searching first")
        elif entry["type"] == "text_without_tool":
            print(f"\n[step {entry['step']}] (model replied in plain text, redirected to use a tool)")


def print_fix(fix: dict):
    print("\n" + "=" * 60)
    print("PROPOSED FIX")
    print("=" * 60)
    print(f"File:       {fix['file']}")
    print(f"Symbol:     {fix['symbol']}")
    print(f"Confidence: {fix['confidence']}")
    print(f"\nWhy:\n{fix['explanation']}")
    print(f"\n--- ORIGINAL ---\n{fix['original_code']}")
    print(f"\n--- PROPOSED FIX ---\n{fix['fixed_code']}")
    print("\n(This is a PROPOSAL only - review it yourself before applying it. "
          "Stage 4 does not edit files on disk.)")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Agent that investigates a bug and proposes a fix.")
    parser.add_argument("bug_description", help="Plain-English description of the bug, in quotes.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                         help=f"Max reasoning/tool-call steps before giving up (default {DEFAULT_MAX_STEPS}).")
    args = parser.parse_args()

    client, collection, bm25, chunks = setup()
    fix, transcript = run_agent(client, collection, bm25, chunks, args.bug_description, args.max_steps)

    print_transcript(transcript)
    if fix:
        print_fix(fix)
    else:
        print(f"\nAgent did not reach a proposed fix within {args.max_steps} steps. "
              f"Try increasing --max-steps or rephrasing the bug description.")


if __name__ == "__main__":
    main()