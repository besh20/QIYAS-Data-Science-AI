"""
Stage 1: Code Explainer / Onboarding Risk Assessor
----------------------------------------------------
Takes a source file (optionally, a specific function name inside it),
and returns a structured assessment: summary, explanation, edge cases,
a risk_level with reasoning, and a suggested docstring.

Why structured output instead of free text?
A plain paragraph is easy to generate but hard to use programmatically
(can't filter by risk, can't build a dashboard, can't chain into later
stages). Forcing a JSON schema via tool-calling means the model MUST
return exactly the fields we need, in a predictable shape - which is
what makes this composable with Stage 2+ (RAG over a whole repo) later.

Usage:
    python explain.py sample_code/example_functions.py
    python explain.py sample_code/example_functions.py --function get_user_data
"""

import argparse
import ast
import json
import os
import sys

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Model name kept as a single config variable on purpose - Groq's free
# catalog changes without warning (it dropped two models in Aug 2026),
# so if this model disappears, swap the string here rather than hunting
# through the file.
MODEL = "openai/gpt-oss-120b"

# The schema we force the model to fill in. This is the core design
# decision of this stage: risk_level exists because a new hire onboarding
# onto a codebase doesn't just want "what does this do" - they want
# "what should I be careful about before I touch this."
#
# Groq's API is OpenAI-compatible, so tools are described in OpenAI's
# function-calling format (a "function" object), not Anthropic's
# "input_schema" format.
EXPLAIN_TOOL = {
    "type": "function",
    "function": {
        "name": "report_code_analysis",
        "description": "Report a structured analysis of a piece of source code.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One-line, plain-English summary of what this code does."
                },
                "explanation": {
                    "type": "string",
                    "description": "A clear paragraph explaining how the code works, step by step."
                },
                "edge_cases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Concrete inputs or conditions that could break this code "
                                    "(empty input, None, concurrency, huge input, etc.)."
                },
                "risk_level": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "How risky it is for a new developer to modify this code "
                                    "without deep testing."
                },
                "risk_reasoning": {
                    "type": "string",
                    "description": "Why this risk level was chosen - e.g. global state, "
                                    "no error handling, mutable defaults, SQL injection, etc."
                },
                "suggested_docstring": {
                    "type": "string",
                    "description": "A ready-to-use docstring for this function, in standard "
                                    "Python docstring style."
                }
            },
            "required": [
                "summary", "explanation", "edge_cases",
                "risk_level", "risk_reasoning", "suggested_docstring"
            ]
        }
    }
}

SYSTEM_PROMPT = """You are a senior engineer helping a new hire onboard onto an \
unfamiliar codebase. You review code and flag real risks a newcomer would miss - \
things like global mutable state, missing error handling, mutable default \
arguments, unguarded recursion, SQL injection, or silent failure modes. \
Be specific and concrete, not generic. Always call the report_code_analysis tool \
with your findings - never respond in plain text."""


def extract_function_source(file_path: str, function_name: str) -> str:
    """Pull just one function's source out of a file using the ast module,
    instead of naive string search - this handles nested functions, decorators,
    and multi-line signatures correctly."""
    with open(file_path, "r") as f:
        source = f.read()

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return ast.get_source_segment(source, node)

    raise ValueError(f"Function '{function_name}' not found in {file_path}")


def analyze_code(client: Groq, code: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1500,
        tools=[EXPLAIN_TOOL],
        tool_choice={"type": "function", "function": {"name": "report_code_analysis"}},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this code:\n\n```python\n{code}\n```"},
        ],
    )

    message = response.choices[0].message
    if message.tool_calls:
        call = message.tool_calls[0]
        if call.function.name == "report_code_analysis":
            return json.loads(call.function.arguments)

    raise RuntimeError("Model did not return the expected tool call.")


def print_report(code: str, report: dict):
    risk_color = {"low": "\033[92m", "medium": "\033[93m", "high": "\033[91m"}
    reset = "\033[0m"
    color = risk_color.get(report["risk_level"], "")

    print("\n" + "=" * 60)
    print("SUMMARY:", report["summary"])
    print("=" * 60)
    print(f"\nRISK LEVEL: {color}{report['risk_level'].upper()}{reset}")
    print("Why:", report["risk_reasoning"])
    print("\nEXPLANATION:")
    print(report["explanation"])
    print("\nEDGE CASES:")
    for ec in report["edge_cases"]:
        print(f"  - {ec}")
    print("\nSUGGESTED DOCSTRING:")
    print(report["suggested_docstring"])
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Explain and risk-assess a piece of code.")
    parser.add_argument("file", help="Path to a Python source file.")
    parser.add_argument("--function", help="Name of a specific function to analyze. "
                                            "If omitted, the whole file is analyzed.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of formatted output.")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Set GROQ_API_KEY in your .env file (see .env.example).")
        print("Get a free key at https://console.groq.com/keys")
        sys.exit(1)

    client = Groq(api_key=api_key)

    if args.function:
        code = extract_function_source(args.file, args.function)
    else:
        with open(args.file, "r") as f:
            code = f.read()

    report = analyze_code(client, code)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(code, report)


if __name__ == "__main__":
    main()
