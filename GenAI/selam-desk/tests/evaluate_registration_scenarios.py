"""Evaluates the registration state machine across several scripted scenarios:
clean success, a user who makes mistakes, a user who cancels, and a user who
gives an invalid value and then corrects it. Runs OFFLINE (mocks extraction)
so it works without internet/API. This produces the "task completion rate"
and "wrong-action rate" numbers for your evaluation section.

Run: python tests/evaluate_registration_scenarios.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch
from src.agent import RegistrationSession

# each scenario: (name, list of user turns, expected final status)
SCENARIOS = [
    ("clean_success",
     ["Abebe Kebede", "0911223344", "Computer Science", "2015-03-10", "none", "yes"],
     "saved"),

    ("bad_phone_then_fix",
     ["Almaz Tesfaye", "12345", "0922334455", "Law", "2012-07-01", "none", "yes"],
     "saved"),

    ("bad_date_then_fix",
     ["Kebede Alemu", "0933445566", "Medicine", "2015-14-40", "2013-05-20", "none", "yes"],
     "saved"),

    ("user_cancels_midway",
     ["Sara Tadesse", "cancel"],
     "cancelled"),

    ("user_rejects_summary_then_redoes",
     ["Yonas Girma", "0944556677", "Engineering", "2010-01-15", "none", "no",
      "Yonas G. Girma", "0944556677", "Engineering", "2010-01-15", "none", "yes"],
     "saved"),
]


def fake_extract_field(user_text, field_name, field_note, field_type="text"):
    return user_text


def run_scenario(name, turns, expected_final):
    session = RegistrationSession()
    statuses = []
    with patch("src.agent.extract_field", side_effect=fake_extract_field):
        for turn in turns:
            status, msg = session.submit_answer(turn)
            statuses.append(status)
    final = statuses[-1]
    passed = (final == expected_final)
    return passed, statuses, session.collected if final == "saved" else None


def main():
    results = []
    for name, turns, expected in SCENARIOS:
        passed, statuses, collected = run_scenario(name, turns, expected)
        results.append((name, passed, statuses, collected))
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {name}: final={statuses[-1]} (expected {expected})")
        if collected:
            print(f"       saved record: {collected}")

    n_total = len(results)
    n_passed = sum(1 for r in results if r[1])
    completion_rate = n_passed / n_total

    # "wrong action" check: for saved scenarios, did an invalid value ever
    # slip through into the final saved record? (it shouldn't -- validators
    # should have caught it before it was ever stored)
    wrong_actions = 0
    for name, passed, statuses, collected in results:
        if collected:
            phone = collected.get("phone", "")
            if phone and not (phone.startswith("+251") and len(phone) == 13):
                wrong_actions += 1

    print(f"\nTask completion rate: {n_passed}/{n_total} = {completion_rate:.0%}")
    print(f"Wrong-action rate (invalid data reaching DB): {wrong_actions}/{n_total} = {wrong_actions/n_total:.0%}")

    report_path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_report_registration.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Registration Flow -- Scenario Evaluation\n\n")
        f.write(f"- Task completion rate: **{n_passed}/{n_total} = {completion_rate:.0%}**\n")
        f.write(f"- Wrong-action rate: **{wrong_actions}/{n_total} = {wrong_actions/n_total:.0%}**\n\n")
        f.write("| Scenario | Result | Final status |\n|---|---|---|\n")
        for name, passed, statuses, _ in results:
            f.write(f"| {name} | {'PASS' if passed else 'FAIL'} | {statuses[-1]} |\n")

    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
