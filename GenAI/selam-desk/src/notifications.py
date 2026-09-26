"""Real staff notification via a free webhook (Slack or Discord incoming
webhook both work -- both are free to set up). If no webhook is configured,
this degrades to a no-op so the app still runs without one; it never
crashes the conversation over a notification failing.

Setup (either works, both free):
- Discord: Server Settings -> Integrations -> Webhooks -> New Webhook -> copy URL
- Slack: api.slack.com/messaging/webhooks -> create an Incoming Webhook -> copy URL
Then set STAFF_WEBHOOK_URL in .streamlit/secrets.toml (or Streamlit Cloud secrets).
"""
import requests

_webhook_url = None


def configure_webhook(url: str | None):
    global _webhook_url
    _webhook_url = url or None


def is_configured() -> bool:
    return _webhook_url is not None


def notify(event_type: str, summary: str, ref_id: int | None = None) -> bool:
    """Fire-and-forget notification. Returns True if it was actually sent,
    False if skipped (not configured) or failed (network issue, etc.) --
    callers should never treat a False as a reason to fail the request.
    """
    if not _webhook_url:
        return False
    label = {"escalation": "🚨 ESCALATION", "complaint": "📝 Complaint"}.get(event_type, event_type)
    text = f"**{label}** (ref {ref_id})\n{summary}"
    payload = {"content": text} if "discord" in _webhook_url else {"text": text}
    try:
        resp = requests.post(_webhook_url, json=payload, timeout=5)
        return resp.status_code < 300
    except requests.RequestException:
        return False  # network hiccup -- don't let this break the conversation
