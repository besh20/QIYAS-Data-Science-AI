# Selam Desk

A bilingual (Amharic/English) tool-calling AI agent for a university
registration desk — built as an Intelligent Data/AI Engineering capstone
project, grounded in an Ethiopian context.

## Why this project
Registration day at Ethiopian universities is chaotic: students queue to ask
the same handful of questions (fees, requirements, deadlines), register for
programs, and occasionally need to report a problem (wrong grades, lost ID).
Selam Desk is a receptionist **agent** that handles all three — not a
scripted FAQ bot, but a system where an LLM decides, turn by turn, which
action a student actually needs and calls the right tool for it.

## What it does
- Answers factual questions by retrieving from the university's own
  documents (RAG), in whichever language/script the student uses
- Registers a student: collects details conversationally, validates
  Ethiopian phone numbers and **Ethiopian-calendar** birth dates, shows a
  confirmation preview, and only saves after explicit confirmation
- Logs complaints (e.g. "my grades are wrong") for staff follow-up
- Escalates to a human when it can't or shouldn't handle something itself,
  with an optional real-time notification to a Slack/Discord channel

## Architecture
1. **Persistent chat session** (`src/agent.py`, `client.chats.create`) — the
   model sees the full conversation, so follow-ups like "what can you help
   me with?" -> "information" -> "what are the requirements?" work
   naturally, with no hand-built state machine.
2. **Five tools the model decides when to call**, each a plain Python
   function with a docstring the model reads to know when/how to use it:
   `search_knowledge_base`, `preview_registration`/`submit_registration`,
   `file_complaint`, `escalate_to_staff`.
3. **Per-session tool isolation** — `build_tools()` creates fresh closures
   per chat session, so two simultaneous students on the live deployment
   can never confirm each other's registration.
4. **Classical ML baseline** (`src/intent.py`, TF-IDF + Logistic Regression)
   — kept as an evaluation baseline for the course-relevant comparison, not
   used for live routing (the agent's own reasoning replaces that role).
5. **RAG** (`src/rag.py`) — `multilingual-e5-small` embeddings + FAISS over
   `data/knowledge_base.md`.
6. **DB** (`src/db.py`) — SQLite: registrations, complaints, escalations.
7. **Prompt-injection defense** (`src/guardrails.py`) — flags role-override
   attempts for logging and reminds the model of its rules on every turn.
   The real defense-in-depth is unchanged regardless: every tool validates
   its own arguments before touching the DB, so even a successful injection
   can't write bad data.
8. **Staff notifications** (`src/notifications.py`) — a free Slack/Discord
   webhook, so escalations/complaints reach an actual human, not just a table.
9. **Structured logging + rate limit** (`src/logging_utils.py`, `app.py`) —
   every message and tool call logged as JSON lines to `data/app.log`; each
   session capped at 40 messages to protect the shared free API quota.

## Project structure
```
selam-desk/
├── app.py                  # Streamlit UI
├── src/
│   ├── agent.py             # chat session, the 5 tools, retry/fallback
│   ├── rag.py                # embeddings + FAISS retrieval
│   ├── intent.py              # TF-IDF baseline classifier (evaluation only)
│   ├── db.py                   # SQLAlchemy models + save functions
│   ├── validators.py            # phone / name / Ethiopian date validation
│   ├── guardrails.py             # prompt-injection heuristics
│   ├── logging_utils.py           # structured JSON-lines logging
│   └── notifications.py            # Slack/Discord webhook
├── data/
│   ├── questions.csv          # intent-classifier training data
│   ├── eval_questions.csv      # HELD-OUT intent-classifier test data
│   ├── knowledge_base.md        # RAG source documents
│   └── registration_fields.json  # form field definitions
└── tests/
    ├── test_tools.py            # offline, no internet needed
    ├── evaluate_intent.py         # TF-IDF baseline on held-out data
    ├── evaluate_retrieval.py        # RAG retrieval accuracy
    └── evaluate_agent_live.py         # real conversations through the live agent
```

## IMPORTANT — data provenance
`data/questions.csv`, `data/knowledge_base.md`, and `data/eval_questions.csv`
currently contain **synthetic placeholder data**, written to develop and
test the pipeline before real university data was available. State this
explicitly in the presentation: it is not a claim about real accuracy, and
the intent classifier's ~62% held-out accuracy is a synthetic-data baseline,
not a production number. Swap in real registration-day questions and the
real university handbook before final submission if possible — the exact
same scripts (`evaluate_intent.py`, `evaluate_retrieval.py`) then produce
honest, real numbers with zero code changes.

## Setup
```bash
pip install -r requirements.txt
```
Get a free Gemini API key at https://aistudio.google.com/apikey

**API key design: every visitor brings their own.** There is no shared
app-wide key. The Streamlit sidebar requires each visitor to paste their
own free Gemini key before the chat unlocks — this means the app has no
shared rate-limit pool and no cost exposure for you as the deployer,
regardless of how many people use it. For local development convenience
only, you can put your own key in `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-key-here"
```
This only pre-fills the sidebar field on YOUR machine while testing. It is
never read on the deployed app unless you also set it as a Streamlit Cloud
secret (which you shouldn't, for a real deployment) — leaving it unset
there means every visitor, including you in production, must enter their
own key. **Never commit `.streamlit/secrets.toml` to GitHub.**

### Staff webhook (optional, free, ~1 minute) — this ONE is the app owner's
Unlike the Gemini key, `STAFF_WEBHOOK_URL` is yours, not the visitor's — it
points at your own staff Slack/Discord channel, so it's fine (and necessary)
to set this as a real Streamlit Cloud secret if you want live notifications.
- **Discord:** Server Settings → Integrations → Webhooks → New Webhook → copy the URL
- **Slack:** api.slack.com/messaging/webhooks → create an Incoming Webhook → copy the URL

```toml
STAFF_WEBHOOK_URL = "your-webhook-url-here"
```
Without it, the app still runs fine — escalations/complaints are still
logged to the DB, just not pushed to a live channel.

## Run
```bash
streamlit run app.py
```

## Deploying to Streamlit Community Cloud
1. Push this folder to a GitHub repo. Add `.streamlit/` to `.gitignore` —
   do not commit it; visitors don't need it since they bring their own key.
2. At share.streamlit.io, sign in with GitHub, "New app", pick the repo,
   set the main file to `app.py`.
3. In the deployed app's Settings → Secrets, optionally set
   `STAFF_WEBHOOK_URL` (your own channel) — do NOT set `GEMINI_API_KEY`
   there, so every visitor is required to enter their own in the sidebar.
4. Deploy, then paste your own key in the sidebar and run through info /
   registration / complaint / escalation once on the live URL before presenting.

**Why BYO-key is the right design here:** with a shared app key, the free
daily quota would be one pool split across every visitor, and you'd bear
all the cost/rate-limit risk. With each visitor supplying their own key,
there's no shared quota to exhaust and no cost exposure to you at all — the
app scales to any number of visitors with zero additional risk on your end.

## Testing
- `python tests/test_tools.py` — offline, no internet needed, tests the
  five tools directly (validation, confirm-before-save, session isolation).
- `python tests/evaluate_intent.py` — TF-IDF baseline accuracy on held-out data.
- `python tests/evaluate_retrieval.py` — RAG retrieval accuracy (needs
  internet once to download the embedding model, then works offline).
- `python tests/evaluate_agent_live.py` — runs real scripted conversations
  through the actual live agent (needs internet + API key).

## Known limitations (production-readiness gaps)
Presented openly rather than hidden — a maturity slide, not a weakness:
- No student authentication — anyone can register as anyone
- SQLite resets on every Streamlit Cloud redeploy; a real deployment needs
  a hosted DB (e.g. free-tier Postgres on Supabase)
- No cost controls beyond the per-session message cap; a public deployment
  would need per-IP/day limits too
- Prompt-injection defense is heuristic, not a hard guarantee — the DB-write
  validation is the actual safety boundary, not the heuristic
- `program` is free text, not validated against a real program list
- BYO-key design means a visitor with no Gemini account/key can't use the
  app at all — a reasonable trade-off for a free demo, worth mentioning
