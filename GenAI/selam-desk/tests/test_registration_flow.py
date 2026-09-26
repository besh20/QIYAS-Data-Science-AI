"""Tests the registration state machine end-to-end WITHOUT calling Gemini or
needing internet -- extract_field is mocked to return scripted answers.
Run: python tests/test_registration_flow.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch
from src.agent import RegistrationSession

# scripted user answers, in the order fields are asked
# (full_name, phone, program, date_of_birth, id_number)
SCRIPTED_ANSWERS = [
    "Abebe Kebede",
    "0911223344",
    "Computer Science",
    "2015-03-10",
    "no id",  # optional field, extraction will likely return NONE
]


def fake_extract_field(user_text, field_name, field_note, field_type="text"):
    # just echo back what the "user" typed -- this simulates a perfect LLM
    return user_text


def run_happy_path():
    print("=== Happy path (valid answers, confirm yes) ===")
    session = RegistrationSession()
    with patch("src.agent.extract_field", side_effect=fake_extract_field):
        for answer in SCRIPTED_ANSWERS:
            status, msg = session.submit_answer(answer)
            print(f"  [{status}] {msg}")
            if status == "confirm":
                break
        # confirm
        status, msg = session.submit_answer("yes")
        print(f"  [{status}] {msg}")
    assert status == "saved", "Expected registration to save"
    print("PASSED\n")


def run_invalid_phone():
    print("=== Invalid phone number should be rejected ===")
    session = RegistrationSession()
    with patch("src.agent.extract_field", side_effect=fake_extract_field):
        session.submit_answer("Abebe Kebede")
        status, msg = session.submit_answer("12345")  # bad phone
        print(f"  [{status}] {msg}")
    assert status is False, "Expected phone validation to fail"
    print("PASSED\n")


def run_invalid_ethiopian_date():
    print("=== Invalid Ethiopian date should be rejected ===")
    session = RegistrationSession()
    with patch("src.agent.extract_field", side_effect=fake_extract_field):
        session.submit_answer("Abebe Kebede")
        session.submit_answer("0911223344")
        session.submit_answer("Computer Science")
        status, msg = session.submit_answer("2015-14-40")  # bad month/day
        print(f"  [{status}] {msg}")
    assert status is False, "Expected date validation to fail"
    print("PASSED\n")


def run_cancel():
    print("=== Cancel mid-flow ===")
    session = RegistrationSession()
    with patch("src.agent.extract_field", side_effect=fake_extract_field):
        session.submit_answer("Abebe Kebede")
        status, msg = session.submit_answer("cancel")
        print(f"  [{status}] {msg}")
    assert status == "cancelled"
    print("PASSED\n")


if __name__ == "__main__":
    run_happy_path()
    run_invalid_phone()
    run_invalid_ethiopian_date()
    run_cancel()
    print("All registration flow tests passed.")
