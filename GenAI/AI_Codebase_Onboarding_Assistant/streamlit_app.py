"""
Streamlit UI for the AI Codebase Onboarding Assistant
----------------------------------------------------------
This is a THIN VISUAL LAYER over the exact same code the CLI tools use -
it imports and calls explain.py, ask.py, agent.py, apply_fix.py, eval.py,
and retrieval.py directly. Nothing is reimplemented here; every claim
made in the README about retrieval accuracy, agent behavior, etc. applies
identically whether you use the terminal or this UI.

Run with:
    streamlit run streamlit_app.py
"""

import difflib
import json
import os

import chromadb
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

import agent as agent_module
import apply_fix as apply_fix_module
import ask as ask_module
import eval as eval_module
import explain as explain_module
import ingest as ingest_module
from retrieval import build_bm25_index

load_dotenv()

DB_PATH = "./chroma_db"
COLLECTION_NAME = "codebase"

st.set_page_config(page_title="AI Codebase Onboarding Assistant", layout="wide")


# ----------------------------------------------------------------------
# Shared setup - cached so re-running a tab doesn't reload everything
# ----------------------------------------------------------------------

def get_groq_client():
    """Not cached with @st.cache_resource on purpose - the key can come
    from a different source per user (sidebar input takes priority),
    so caching it globally would leak one visitor's key into everyone
    else's session on a shared deployment."""
    secrets_key = ""
    try:
        secrets_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass  # no secrets.toml configured (normal for local runs) - not an error

    api_key = (
        st.session_state.get("user_groq_key", "").strip()
        or secrets_key
        or os.environ.get("GROQ_API_KEY", "")
    )
    if not api_key:
        return None
    return Groq(api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_index():
    """Returns (collection, bm25, chunks) or (None, None, None) if not indexed yet."""
    if not os.path.isdir(DB_PATH):
        return None, None, None
    chunks_path = os.path.join(DB_PATH, "chunks.json")
    if not os.path.isfile(chunks_path):
        return None, None, None
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    bm25 = build_bm25_index(chunks)
    return collection, bm25, chunks


def require_setup():
    """Call at the top of any tab that needs the index + API key.
    Shows a clear message and halts that tab's rendering if not ready -
    but doesn't crash the whole app (unlike the CLI's sys.exit)."""
    client = get_groq_client()
    if client is None:
        st.error("No Groq API key found. Enter your own free key in the sidebar, "
                  "or (if running locally) add GROQ_API_KEY to your .env file.")
        st.stop()

    collection, bm25, chunks = get_index()
    if collection is None:
        st.error("No index found. Go to the 'Index' tab and click 'Build / rebuild index' first.")
        st.stop()

    return client, collection, bm25, chunks


# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------

st.title("AI Codebase Onboarding Assistant")
st.caption("Same engine as the CLI (explain.py / ask.py / agent.py / apply_fix.py / eval.py) - this is a UI on top of it, not a separate implementation.")

with st.sidebar:
    st.markdown("### Groq API key")
    st.text_input(
        "Your own key (free, no card)",
        type="password",
        key="user_groq_key",
        placeholder="gsk_...",
        help="Get one free at console.groq.com/keys. Entered here, it stays in "
             "your browser session only - never saved to disk or shown to other visitors.",
    )
    st.caption("If left blank, falls back to this server's own key (if configured) - "
               "using your own key means you're never sharing someone else's rate limit.")

tab_index, tab_explain, tab_ask, tab_agent, tab_eval = st.tabs(
    ["Index", "Explain Code", "Ask", "Investigate & Fix", "Retrieval Eval"]
)

# --- Index tab ---
with tab_index:
    st.subheader("Repo indexing")
    collection, bm25, chunks = get_index()
    if collection is not None:
        st.success(f"Index found: {len(chunks)} chunks indexed.")
    else:
        st.warning("No index found yet.")

    repo_path = st.text_input("Repo path to index", value=".")
    if st.button("Build / rebuild index"):
        with st.spinner("Indexing (first run downloads a small local embedding model)..."):
            ingest_module.build_index(repo_path)
        get_index.clear()
        st.success("Indexed. Switch tabs or click again to refresh the count above.")

# --- Explain Code tab ---
with tab_explain:
    st.subheader("Explain & risk-assess a piece of code")
    client = get_groq_client()
    if client is None:
        st.error("No Groq API key found. Enter your own free key in the sidebar, "
                  "or (if running locally) add GROQ_API_KEY to your .env file.")
    else:
        code_input = st.text_area(
            "Paste a Python function or file",
            height=200,
            placeholder="def example(x):\n    return x / 0",
        )
        if st.button("Analyze", key="explain_button") and code_input.strip():
            with st.spinner("Analyzing..."):
                report = explain_module.analyze_code(client, code_input)

            risk_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}
            st.markdown(f"### {risk_color.get(report['risk_level'], '')} Risk: {report['risk_level'].upper()}")
            st.write(report["risk_reasoning"])
            st.markdown("**Summary**")
            st.write(report["summary"])
            st.markdown("**Explanation**")
            st.write(report["explanation"])
            st.markdown("**Edge cases**")
            for ec in report["edge_cases"]:
                st.write(f"- {ec}")
            st.markdown("**Suggested docstring**")
            st.code(report["suggested_docstring"])

# --- Ask tab ---
with tab_ask:
    st.subheader("Ask a question about the indexed repo")
    question = st.text_input("Question", placeholder="where is risk assessed in this codebase?")
    col1, col2 = st.columns(2)
    with col1:
        mode = st.selectbox("Retrieval mode", ["hybrid", "vector", "keyword"], index=0)
    with col2:
        k = st.slider("Chunks to retrieve (k)", 1, 10, 4)

    if st.button("Ask", key="ask_button") and question.strip():
        client, collection, bm25, chunks = require_setup()
        with st.spinner("Retrieving and answering..."):
            retrieved = ask_module.retrieve(collection, bm25, chunks, question, k, mode)
            report = ask_module.answer_question(client, question, retrieved)

        st.markdown(f"**Grounded:** {'✅ Yes' if report['grounded'] else '⚠️ No - the model said the retrieved code did not fully answer this'}")
        st.markdown("### Answer")
        st.write(report["answer"])

        st.markdown("### Citations")
        for c in report["citations"]:
            st.write(f"- `{c['file']} :: {c['symbol']}` — {c['why_relevant']}")

        with st.expander(f"Retrieved chunks ({len(retrieved)}) - what the model was allowed to see"):
            for r in retrieved:
                st.markdown(f"**{r['file']} :: {r['symbol']}** (lines {r['start_line']}-{r['end_line']})")
                st.code(r["code"], language="python")

# --- Investigate & Fix tab ---
with tab_agent:
    st.subheader("Describe a bug - the agent investigates and proposes a fix")
    bug_description = st.text_area(
        "Bug description",
        placeholder="Users report that a malicious user_id crashes the app or leaks data",
    )
    max_steps = st.slider("Max agent steps", 2, 10, 6)

    if st.button("Investigate", key="agent_button") and bug_description.strip():
        client, collection, bm25, chunks = require_setup()
        with st.spinner("Agent is searching and reasoning (this can take a few steps)..."):
            fix, transcript = agent_module.run_agent(client, collection, bm25, chunks, bug_description, max_steps)

        st.markdown("### Agent transcript")
        for entry in transcript:
            if entry["type"] == "search":
                with st.expander(f"🔍 Step {entry['step']}: searched \"{entry['query']}\""):
                    for r in entry["results"]:
                        st.write(f"- `{r['file']} :: {r['symbol']}` (lines {r['start_line']}-{r['end_line']})")
            elif entry["type"] == "rejected_premature_fix":
                st.warning(f"Step {entry['step']}: blocked a fix proposal made without searching first.")
            elif entry["type"] == "text_without_tool":
                st.info(f"Step {entry['step']}: model replied in plain text, redirected to use a tool.")

        if fix:
            st.session_state["pending_fix"] = fix
            st.session_state["pending_bug_description"] = bug_description
            st.markdown("### Proposed fix")
            st.write(f"**File:** `{fix['file']}` &nbsp;&nbsp; **Symbol:** `{fix['symbol']}` &nbsp;&nbsp; **Confidence:** {fix['confidence']}")
            st.write(fix["explanation"])
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Original**")
                st.code(fix["original_code"], language="python")
            with col_b:
                st.markdown("**Proposed fix**")
                st.code(fix["fixed_code"], language="python")
        else:
            st.error("Agent did not reach a proposed fix. Try increasing max steps or rephrasing the bug description.")
            st.session_state.pop("pending_fix", None)

    # Apply step - only shown once a fix exists in session state, same
    # verify -> diff -> confirm -> write flow as apply_fix.py's CLI.
    if "pending_fix" in st.session_state:
        st.divider()
        st.markdown("### Apply this fix?")
        st.caption("Note: on a public/shared deployment, this writes to that server's "
                   "own ephemeral copy of the code - not your real GitHub repo, and it "
                   "resets on the next redeploy. It's only meaningful when running locally.")
        fix = st.session_state["pending_fix"]

        verified = apply_fix_module.find_original_in_file(fix["file"], fix["original_code"])
        if not verified:
            st.error("Safety check FAILED: the code the agent thinks it saw no longer matches the real file. Refusing to apply.")
        else:
            st.success("Safety check passed: original_code confirmed present in the file.")
            diff = "".join(difflib.unified_diff(
                fix["original_code"].strip().splitlines(keepends=True),
                fix["fixed_code"].strip().splitlines(keepends=True),
                fromfile=f"{fix['file']} (original)",
                tofile=f"{fix['file']} (proposed)",
            ))
            st.code(diff, language="diff")

            col_apply, col_discard = st.columns(2)
            with col_apply:
                if st.button("✅ Apply to disk + draft PR"):
                    apply_fix_module.apply_replacement(fix["file"], fix["original_code"], fix["fixed_code"])
                    pr_text = apply_fix_module.draft_pr_description(fix, st.session_state["pending_bug_description"])
                    pr_path = f"PR_{fix['symbol']}.md"
                    with open(pr_path, "w", encoding="utf-8") as f:
                        f.write(pr_text)
                    st.success(f"Applied. {fix['file']} updated, PR description saved to {pr_path}.")
                    del st.session_state["pending_fix"]
            with col_discard:
                if st.button("❌ Discard"):
                    del st.session_state["pending_fix"]
                    st.info("Discarded. Nothing was changed.")
                    st.rerun()

# --- Retrieval Eval tab ---
with tab_eval:
    st.subheader("Retrieval evaluation (vector vs. keyword vs. hybrid)")
    k_eval = st.slider("k for evaluation", 1, 20, 4, key="eval_k")

    if st.button("Run evaluation"):
        collection, bm25, chunks = get_index()
        if collection is None:
            st.error("No index found. Build one in the 'Index' tab first.")
        elif not os.path.isfile("eval_set.json"):
            st.error("No eval_set.json found in this folder.")
        else:
            with open("eval_set.json", "r", encoding="utf-8") as f:
                eval_cases = json.load(f)

            from retrieval import vector_search, keyword_search, hybrid_search
            modes = {
                "vector": lambda q, k: vector_search(collection, q, k),
                "keyword": lambda q, k: keyword_search(bm25, chunks, q, k),
                "hybrid": lambda q, k: hybrid_search(collection, bm25, chunks, q, k),
            }
            with st.spinner("Evaluating..."):
                results = [eval_module.evaluate_mode(name, fn, eval_cases, k_eval) for name, fn in modes.items()]

            st.markdown("### Results")
            chart_data = {r["mode"]: r["hit_rate"] * 100 for r in results}
            st.bar_chart(chart_data)

            for r in results:
                st.write(f"**{r['mode']}** — Hit Rate: {r['hit_rate']:.1%}, MRR: {r['mrr']:.3f}, Hits: {r['hits']}/{r['total']}")

            with st.expander("Misses per mode"):
                for r in results:
                    misses = [q for q in r["per_question"] if q["found_at_rank"] is None]
                    if misses:
                        st.write(f"**{r['mode']}:**")
                        for m in misses:
                            st.write(f"  - \"{m['question']}\" (expected {m['expected']})")
                    else:
                        st.write(f"**{r['mode']}:** no misses at k={k_eval}")
