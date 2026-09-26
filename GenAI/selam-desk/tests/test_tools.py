"""Tests the five agent tools directly as plain Python functions -- no LLM,
no internet, no mocking needed, since build_tools() returns real callables.
This tests the SAFETY property (no save without confirmed preview, no
invalid data reaches the DB) independent of whether the model behaves well.

Run: python tests/test_tools.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent import build_tools
from src import db as db_mod


def fresh_tools():
    search, preview, submit, complaint, escalate = build_tools()
    return preview, submit, complaint, escalate


def test_happy_path():
    print("=== Happy path: preview then confirmed submit ===")
    preview, submit, _, _ = fresh_tools()
    result = preview("Abebe Kebede", "0911223344", "Computer Science", "2015-03-10")
    print(" preview:", result.splitlines()[0])
    assert "Preview OK" in result

    result = submit("Abebe Kebede", "0911223344", "Computer Science", "2015-03-10")
    print(" submit:", result)
    assert "Saved successfully" in result
    print("PASSED\n")


def test_submit_without_preview_is_rejected():
    print("=== Submit without a prior preview must be rejected ===")
    _, submit, _, _ = fresh_tools()  # fresh session_state -- no preview ever happened
    result = submit("Nobody Nowhere", "0911223344", "Law", "2015-03-10")
    print(" submit:", result)
    assert result.startswith("ERROR")
    print("PASSED\n")


def test_submit_with_mismatched_data_is_rejected():
    print("=== Submit with different data than the preview must be rejected ===")
    preview, submit, _, _ = fresh_tools()
    preview("Abebe Kebede", "0911223344", "Computer Science", "2015-03-10")
    # model tries to submit DIFFERENT details than what was previewed/confirmed
    result = submit("Different Person", "0922334455", "Law", "2010-01-01")
    print(" submit:", result)
    assert result.startswith("ERROR")
    print("PASSED\n")


def test_invalid_phone_rejected_at_preview():
    print("=== Invalid phone rejected before it ever reaches a preview ===")
    preview, _, _, _ = fresh_tools()
    result = preview("Abebe Kebede", "12345", "Computer Science", "2015-03-10")
    print(" preview:", result)
    assert result.startswith("ERROR")
    print("PASSED\n")


def test_invalid_date_rejected_at_preview():
    print("=== Invalid Ethiopian date rejected before it ever reaches a preview ===")
    preview, _, _, _ = fresh_tools()
    result = preview("Abebe Kebede", "0911223344", "Computer Science", "2015-14-40")
    print(" preview:", result)
    assert result.startswith("ERROR")
    print("PASSED\n")


def test_complaint_logging():
    print("=== Complaint logging ===")
    _, _, complaint, _ = fresh_tools()
    result = complaint("My grades are showing incorrectly", "Sara Tadesse", "0911111111")
    print(" complaint:", result)
    assert "Complaint logged" in result
    print("PASSED\n")


def test_escalation_logging():
    print("=== Escalation logging ===")
    _, _, _, escalate = fresh_tools()
    result = escalate("Student needs urgent help, outside agent scope", "0922222222")
    print(" escalate:", result)
    assert "Escalation logged" in result
    print("PASSED\n")


def test_two_sessions_dont_leak_previews():
    print("=== Two independent sessions must not share preview state ===")
    preview_a, submit_a, _, _ = fresh_tools()
    preview_b, submit_b, _, _ = fresh_tools()
    preview_a("Person A", "0911111111", "Program A", "2015-01-01")
    # session B never called preview -- its submit must still fail
    result = submit_b("Person A", "0911111111", "Program A", "2015-01-01")
    print(" session B submit (should fail):", result)
    assert result.startswith("ERROR")
    print("PASSED\n")


if __name__ == "__main__":
    if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "selam_desk.db")):
        os.remove(os.path.join(os.path.dirname(__file__), "..", "data", "selam_desk.db"))

    test_happy_path()
    test_submit_without_preview_is_rejected()
    test_submit_with_mismatched_data_is_rejected()
    test_invalid_phone_rejected_at_preview()
    test_invalid_date_rejected_at_preview()
    test_complaint_logging()
    test_escalation_logging()
    test_two_sessions_dont_leak_previews()
    print("All tool tests passed.")
