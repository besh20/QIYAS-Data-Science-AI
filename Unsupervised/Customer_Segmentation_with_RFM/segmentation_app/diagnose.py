"""
Diagnostic script -- checks whether your pandas installation is using
PyArrow-backed columns (the root cause of the Windows crash we've been
chasing), and confirms whether the fix in pipeline.py resolves it.

Run with:
    python -X faulthandler diagnose.py "Online Retail.csv"
"""

import sys
import faulthandler
faulthandler.enable()

import pandas as pd
from pipeline import clean_transactions, _de_arrow


def main(path):
    print(f"pandas version: {pd.__version__}", flush=True)
    try:
        import pyarrow
        print(f"pyarrow version: {pyarrow.__version__}", flush=True)
    except ImportError:
        print("pyarrow: not installed", flush=True)

    print(f"\n[1] Reading file: {path}", flush=True)
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    print(f"[1] OK -- shape: {df.shape}", flush=True)

    print("\n[2] Column dtypes as read (BEFORE any fix):", flush=True)
    for col in df.columns:
        flag = " <-- PyArrow-backed (this is the crash cause)" if "pyarrow" in str(df[col].dtype).lower() or "arrow" in str(df[col].dtype).lower() else ""
        print(f"     {col:15s} {str(df[col].dtype):25s}{flag}", flush=True)

    print("\n[3] Applying _de_arrow() fix from pipeline.py...", flush=True)
    fixed = _de_arrow(df)
    print("[3] Column dtypes AFTER fix:", flush=True)
    for col in fixed.columns:
        print(f"     {col:15s} {str(fixed[col].dtype)}", flush=True)

    print("\n[4] Parsing InvoiceDate with pure-Python datetime.strptime (bypasses PyArrow entirely)...", flush=True)
    from pipeline import _parse_date_safe
    parsed = [_parse_date_safe(v) for v in fixed["InvoiceDate"].tolist()]
    fixed["InvoiceDate"] = pd.Series(parsed, index=fixed.index, dtype="datetime64[ns]")
    print(f"[4] OK -- no crash. dtype: {fixed['InvoiceDate'].dtype}, range: {fixed['InvoiceDate'].min()} to {fixed['InvoiceDate'].max()}", flush=True)

    print("\n[5] Running full clean_transactions() from pipeline.py...", flush=True)
    cleaned = clean_transactions(df)
    print(f"[5] OK -- cleaned shape: {cleaned.shape}", flush=True)

    print("\nALL CHECKS PASSED -- the fix works on your machine.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -X faulthandler diagnose.py "Online Retail.csv"')
        sys.exit(1)
    try:
        main(sys.argv[1])
    except Exception:
        import traceback
        print("\nPYTHON EXCEPTION CAUGHT:", flush=True)
        traceback.print_exc()