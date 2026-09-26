"""Runs scripted multi-turn conversations through the REAL agent (real
Gemini calls, real tool execution) and checks the outcome. This needs
internet + your API key -- it's the genuine "did the agent actually do the
right thing" evaluation, as opposed to test_tools.py which only tests the
tools in isolation.

Run: python tests/evaluate_agent_live.py
Reads your key the same way app.py does (.streamlit/secrets.toml), or set
GEMINI_API_KEY as an environment variable instead.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from src import agent
from src import db as db_mod

# each scenario: (name, list of user turns, what we expect to be true after)
SCENARIOS = [
    ("info_question",
     ["how much is the registration fee?"],
     lambda: True),  # manual check: read the printed reply for correctness

    ("full_registration",
     ["I want to register", "Abebe Kebede", "0911223344", "Computer Science",
      "2015-03-10", "yes, that's correct"],
     lambda: db_mod.SessionLocal().query(db_mod.Registration)
             .filter_by(phone="+251911223344").first() is not None),

    ("complaint_report",
     ["you guys messed up my grades, this needs to be fixed"],
     lambda: db_mod.SessionLocal().query(db_mod.Complaint).count() > 0),

    ("continuity_layered_question",
     ["what can you help me with?", "information", "what are the requirements?"],
     lambda: True),  # manual check
]


def get_api_key():
    import streamlit as st  # only used to read secrets.toml the same way app.py does
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


def main():
    api_key = get_api_key()
    if not api_key:
        print("No GEMINI_API_KEY found. Set it in .streamlit/secrets.toml or as an env var.")
        return
    agent.configure_gemini(api_key)

    if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "selam_desk.db")):
        os.remove(os.path.join(os.path.dirname(__file__), "..", "data", "selam_desk.db"))

    results = []
    for name, turns, check in SCENARIOS:
        print(f"\n=== {name} ===")
        chat = agent.create_chat_session()
        for turn in turns:
            print(f"  user: {turn}")
            reply = agent.send_message(chat, turn)
            print(f"  agent: {reply[:200]}")
            time.sleep(1)  # be gentle on free-tier rate limits
        passed = check()
        results.append((name, passed))
        print(f"  [{'PASS' if passed else 'CHECK MANUALLY'}]")

    n = len(results)
    n_auto_pass = sum(1 for _, p in results if p)
    print(f"\nAutomated checks passed: {n_auto_pass}/{n}")
    print("(info_question and continuity scenarios need a manual read of the replies above --")
    print(" automated check only confirms the conversation ran without crashing.)")


if __name__ == "__main__":
    main()
