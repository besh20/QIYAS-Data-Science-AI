"""Structured JSON-lines logging for observability -- who said what, which
tools fired, and any errors. Writes to data/app.log. This is what lets you
say "here's how we'd monitor this in production" with an actual artifact,
not just a claim.
"""
import json
import os
import time

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "app.log")


def log_event(event_type: str, **fields):
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event_type, **fields}
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass  # never let logging itself break the conversation
