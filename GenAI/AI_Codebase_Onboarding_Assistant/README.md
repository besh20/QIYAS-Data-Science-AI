# AI Codebase Onboarding Assistant

An AI tool that helps a developer get up to speed on an unfamiliar codebase —
built in five stages, each one a real capability on top of the last, in one
growing repo: **explain code → RAG over a repo → hybrid search + measured
retrieval quality → an agent that investigates bugs → an agent that applies
fixes with human approval and drafts a PR.**

Run it on itself: `python ask.py "where is risk assessed in this codebase?"`
or `python agent.py "<a bug description>"` and it reasons about its own code.

---

## What it actually does

Given a bug report in plain English (e.g. *"a malicious user_id crashes the
app or leaks data"*), the assistant:
1. Searches the indexed codebase (hybrid semantic + keyword search) to find
   the actually-relevant code — not just what sounds related.
2. Reasons over what it found, refining its search if the first pass wasn't
   enough.
3. Proposes a concrete fix, grounded in the exact code it retrieved.
4. Shows you a diff, waits for your approval, and only then writes the fix
   to disk and drafts a PR description.

Every step is measurable and inspectable — you can see exactly what was
retrieved, how confident the agent was, and whether its retrieval is
actually accurate (via `eval.py`), not just "seems to work."

## Why this project (and not another PDF chatbot)

Most GenAI portfolio projects retrieve over documents. This one retrieves
over *code*, which has structure (functions, imports, git history) that
plain text-chunking ignores. That structural difference is what makes each
stage a genuine engineering problem — AST-aware chunking, hybrid search,
an agent that has to decide when it has "enough" evidence — instead of a
copy-pasted RAG tutorial with a new dataset.

## How this is different from pasting the bug into ChatGPT/Claude/Cursor

This is a fair question to ask about any AI coding tool, and worth
answering honestly rather than overselling it:

- **Pasting a bug into a chatbot requires you to already know which code is
  relevant** — you have to find and paste it yourself. This tool's whole
  job is doing that step: given only a plain-English symptom, it locates
  the actual code on its own, across an entire repo, not a snippet you
  hand-picked.
- **A generic chatbot has no memory of your codebase's structure** between
  messages unless you keep re-pasting context. This tool has a persistent,
  measured index of the whole repo (`eval.py` reports real Hit Rate/MRR
  numbers — see below) that it reuses across every question and every fix.
- **It's a pipeline you can inspect and improve**, not a black box. You can
  see exactly which chunks were retrieved, swap the retrieval strategy,
  measure whether a change actually improved accuracy, and watch the agent's
  full reasoning transcript — none of which a chat window exposes.
- **Honest limitation**: for a single, already-located bug in a small file,
  pasting into Claude/Cursor directly is often just as fast, sometimes
  faster. This project's value isn't "better than a chatbot at fixing one
  function" — it's demonstrating the *engineering* around retrieval,
  evaluation, and agentic tool use that production AI coding tools (Cursor,
  Copilot Workspace, etc.) are actually built on internally. That's the
  honest pitch: not "a better Cursor," but "understanding what's under
  Cursor's hood, built from scratch."

## Project self-assessment

Rated honestly, not generously — as a learning/portfolio project, not a
production system:

| Metric | /10 | Why |
|---|---|---|
| Relevance to current AI engineering roles | 9 | RAG, retrieval eval, and agentic tool-use are exactly what GenAI/LLM job postings ask for in 2026 |
| Technical difficulty (for a self-taught path) | 8 | AST chunking, hybrid search fusion, multi-turn agent loops, and real failure-mode debugging (token limits, tool-choice non-compliance) go well beyond a single-API-call tutorial |
| Problem-solving demonstrated | 9 | Every stage hit a real bug (Windows path separators, token budget exhaustion, agent search loops, forced-tool-choice failures) that required actual diagnosis, not just following steps |
| Usefulness as a real tool, as-is | 5 | Works well on a small repo; not yet production-ready (no auth, no multi-repo support, free-tier model quality ceiling) |
| Usefulness as a portfolio/interview artifact | 9 | Measured results (not just "it works"), a git history showing real progression, and specific, defensible design decisions to discuss |
| Originality vs. typical bootcamp projects | 8 | Code-focused RAG + a working agent loop is meaningfully less common than another PDF/document chatbot |
| Polish / production-readiness | 4 | No UI, single-repo/single-user, free-tier model, no automated test suite for the tool itself |

**Overall**: strong as a learning project and interview talking point; not
a finished product. That gap is normal and expected at this stage — closing
it (tests, packaging, maybe a UI) is optional future work, not a requirement
for it to be portfolio-ready right now.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY (free, no card - console.groq.com/keys)
python ingest.py .     # index this repo (or point it anywhere)
```

---

## Stage 1: Code Explainer / Risk Assessor

Given a Python file or function, returns a structured analysis: summary,
explanation, edge cases, a risk level (low/medium/high) with reasoning, and
a suggested docstring — via forced JSON schema (tool-calling), not free text.

**Why structured output over free text:** a paragraph is easy to generate
but useless programmatically — you can't filter by risk, chain it into
later stages, or build anything on top of it. Forcing a schema makes the
output composable, which is exactly why Stage 4's agent could later reuse
this same pattern for its own tool calls.

**Why the `risk_level` field:** a new hire onboarding doesn't just want "what
does this do" — they want "what should I be careful about before I touch
this." That reframing, from explainer to risk-assessor, is what makes this
more than a generic "explain this code" demo.

```bash
python explain.py sample_code/example_functions.py --function get_user_data
```

---

## Stage 2: RAG over a whole repo

Indexes an entire codebase into a local vector store (ChromaDB, embeddings
computed locally — free, no rate limit), then answers natural-language
questions grounded in retrieved code, with citations to exact file/line.

**Why AST-aware chunking, not character-count chunking:** naive chunking
(the tutorial default) can slice a function in half, separating logic from
its own definition. Chunking by function/class via Python's `ast` module
guarantees every chunk is a complete, meaningful unit — a real engineering
choice, not a default left untouched.

**Why forced citations + a `grounded` flag:** an answer with no citation is
unverifiable. Forcing every answer to name its source makes retrieval
quality checkable — which is exactly what Stage 3's `eval.py` measures.

**Why local embeddings + API-based generation:** embedding locally is free,
unlimited, and keeps code from leaving your machine during indexing; only
the final answer generation calls an API, which is the one step that
actually needs a strong model.

```bash
python ask.py "where is risk assessed in this codebase?"
```

---

## Stage 3: Hybrid search + retrieval evaluation

Adds keyword (BM25) search alongside vector search, fuses both via
Reciprocal Rank Fusion, and adds `eval.py` — a script that measures, with
real numbers, how often each retrieval mode actually finds the right code.

**Why hybrid, specifically for code:** vector search matches *meaning* and
can miss exact identifiers (a query for `chunk_file` can retrieve a
related-but-wrong function). Keyword search is the opposite — exact but
blind to meaning. This project's own `eval_set.json` has a live example:
keyword-only search ranked `chunk_file` 5th; fusing it with vector search
moved it to 2nd.

**Why measure retrieval at all:** a RAG pipeline can run with zero errors
and still give wrong answers, silently, if it retrieves the wrong chunk.
Hit Rate/MRR turn "I think it works" into a number you can defend.

**Why both Hit Rate and MRR:** Hit Rate only asks "was the right chunk
anywhere in the top-k." MRR rewards ranking it *higher*. In this project's
own data, hybrid tied vector on Hit Rate at k=4 but had a **lower** MRR
(0.323 vs 0.469) — proof hybrid isn't a strict upgrade, it's a genuine
tradeoff depending on k and the fusion weighting.

```bash
python eval.py --k 15 --verbose
```

### Actual eval results (this repo, after fixing a Windows path-separator bug)

```
Mode       Hit Rate     MRR        Hits/Total
vector     100.0%       0.515      8/8
keyword    87.5%        0.255      7/8
hybrid     100.0%       0.369      8/8
```

The bug: `os.path.relpath()` returns backslashes on Windows, but the eval
set used forward slashes, so every question about a file in a subfolder
silently failed the comparison — not a real retrieval failure, a path-
format mismatch. `eval.py` catching this anomaly, rather than accepting a
suspiciously low number, is itself part of the point of having an eval
script.

---

## Stage 4: Tool-calling agent

Takes a plain-English bug description and runs a loop: search the
codebase, look at the result, decide whether to search again or propose a
fix — the model decides how many steps it needs, not a fixed pipeline.

**Why propose-only at this stage, no file edits:** this stage proves out
the *reasoning loop* — the hard part conceptually. Writing to disk is a
separate, smaller addition (Stage 5) bolted on once the loop is trustworthy;
combining both from the start would add risk without adding to what this
stage teaches.

**Why the agent's search tool is Stage 3's `hybrid_search`, unmodified:**
the agent's search quality is only as good as retrieval already measured
in `eval.py`. Reusing the exact same function means every claim about the
agent's retrieval accuracy is backed by real numbers, not a separate,
unmeasured search built just for the agent.

**Why the "must search before proposing" guard rail:** without it, nothing
stops the model from guessing a fix for code it never actually looked at.
The guard rejects any `propose_fix` call made with zero prior searches.

**Why forcing `tool_choice` after 2 searches:** in testing, the free model
repeatedly re-searched — finding the same correct function up to five
times — instead of ever calling `propose_fix`, even with explicit
instructions to stop. Prompting alone didn't fix it, because the issue
wasn't a lack of information, it was the model's own bias toward the
simpler tool. Forcing `tool_choice` at the API level removes that choice
entirely, guaranteeing termination regardless of the model's judgment.
When even the forced model attempted to call the wrong tool anyway (a real,
observed failure — Groq's API rejected it with a 400 error), a fallback
(`finalize_fix`) retries with a short, clean context instead of crashing —
and this fallback was confirmed working in a real run, not just tested in
isolation.

```bash
python agent.py "Users report that a malicious user_id crashes the app or leaks data"
```

---

## Stage 5: Apply the fix + draft a PR

Runs the Stage 4 agent, then — only after a shown diff is confirmed by a
human — writes the fix to the real file and drafts a PR description.
Optionally commits locally to a new branch; never pushes.

**Why verify-before-write:** an LLM can be confidently wrong about what
code it "saw" — it generates plausible tokens, it doesn't literally quote
a buffer. Before writing anything, this checks that `original_code` the
model claims it found still actually exists in the real file. If it
doesn't, the run refuses instead of silently corrupting the file.

**Why confirm, then write, never auto-apply:** an agent that can locate
bugs AND silently rewrite files without a human in the loop is a genuinely
different, much riskier tool than one that only proposes. Every step past
the confirmation prompt is deterministic and reviewable — there's no
"trust me" step.

**Why commit locally at most, never push:** committing and pushing are not
the same risk level. A push can trigger CI, notify teammates, or open a PR
automatically — that should always be a deliberate, separate action a
human runs themselves, never something a script decided on your behalf.

```bash
python apply_fix.py "malicious user_id crashes the app or leaks data"
```

Confirmed working end-to-end in real use: the agent correctly diagnosed and
fixed the SQL-injection vulnerability in `get_user_data`, the safety check
passed, the diff was reviewed and approved, the file was updated, and
`PR_get_user_data.md` was generated — including one real run where the
`finalize_fix` fallback fired mid-agent-loop and recovered successfully.

---

## UI

A Streamlit UI is included as a thin visual layer over the exact same code
the CLI uses (it imports `explain.py`/`ask.py`/`agent.py`/`apply_fix.py`/
`eval.py` directly - nothing is reimplemented). Every tab mirrors a CLI tool:
Index (`ingest.py`), Explain Code (`explain.py`), Ask (`ask.py`),
Investigate & Fix (`agent.py` + `apply_fix.py`, with a proper diff-review +
approve/discard flow), and Retrieval Eval (`eval.py`, with a live chart).

```bash
streamlit run streamlit_app.py
```

## Possible future improvements (not required for this to be "done")
- Multi-repo / multi-language support (currently Python-only, via `ast` -
  a different language needs a different parser, e.g. tree-sitter)
- A stronger (paid) model option for production-quality fix generation

## Tech stack

Python · Groq (`openai/gpt-oss-120b`, free tier) · ChromaDB (local
embeddings) · BM25 (`rank-bm25`) · Reciprocal Rank Fusion · `ast`-based
code chunking · git

## Roadmap

| Stage | Capability | Status |
|---|---|---|
| 1 | Single-function explainer with risk assessment | ✅ done |
| 2 | RAG over a full repo (AST-aware chunking) | ✅ done |
| 3 | Hybrid search + retrieval eval | ✅ done |
| 4 | Tool-calling agent (locate + propose fix) | ✅ done |
| 5 | Apply fix + draft PR | ✅ done |
