"""Evaluates the intent classifier on a HELD-OUT set (data/eval_questions.csv)
that was never used for training. This is what shows real generalization,
not the optimistic 5-fold CV number.

Run: python tests/evaluate_intent.py
Produces: data/eval_report_intent.md (paste numbers/table into your slides)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.intent import train

TRAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "questions.csv")
EVAL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_questions.csv")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_report_intent.md")

LABELS = ["info", "register", "assist", "escalate"]


def main():
    pipe = train()  # always retrain fresh on current questions.csv
    eval_df = pd.read_csv(EVAL_PATH)

    y_true = eval_df["intent"].tolist()
    y_pred = pipe.predict(eval_df["text"]).tolist()

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    print(f"Held-out accuracy: {acc:.3f}  ({len(eval_df)} examples, never seen in training)\n")
    print(report)
    print("Confusion matrix (rows=true, cols=predicted):")
    print("        " + "  ".join(f"{l:>9}" for l in LABELS))
    for label, row in zip(LABELS, cm):
        print(f"{label:>8}" + "  ".join(f"{v:>9}" for v in row))

    # per-language breakdown -- this is the number that matters most for your
    # "Amharic vs English vs code-switched" finding
    print("\nAccuracy by language:")
    lang_lines = []
    for lang, group in eval_df.groupby("language"):
        preds = pipe.predict(group["text"])
        lang_acc = accuracy_score(group["intent"], preds)
        line = f"  {lang}: {lang_acc:.3f}  (n={len(group)})"
        print(line)
        lang_lines.append(line)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Intent Classifier -- Held-out Evaluation\n\n")
        f.write(f"- Held-out accuracy: **{acc:.3f}** ({len(eval_df)} examples)\n")
        f.write(f"- Trained on {len(pd.read_csv(TRAIN_PATH))} examples\n\n")
        f.write("## Accuracy by language\n\n")
        for line in lang_lines:
            f.write(f"- {line.strip()}\n")
        f.write("\n## Classification report\n\n```\n" + report + "\n```\n")
        f.write("\n## Confusion matrix (rows=true, cols=predicted)\n\n")
        f.write("| true \\ pred | " + " | ".join(LABELS) + " |\n")
        f.write("|---" * (len(LABELS) + 1) + "|\n")
        for label, row in zip(LABELS, cm):
            f.write(f"| {label} | " + " | ".join(str(v) for v in row) + " |\n")

    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
