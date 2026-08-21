"""
Run this ONCE to train the segmentation model on your historical data and
save it as segmentation_pipeline.pkl, which the dashboard app loads.

Usage:
    python train_pipeline.py "Online Retail.csv"
"""

import os
import sys
import time
import traceback
import pandas as pd
from pipeline import CustomerSegmentationPipeline, clean_transactions


def main(source_path: str, output_path: str = "segmentation_pipeline.pkl"):
    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"Could not find '{source_path}' in the current folder ({os.getcwd()}). "
            f"Make sure the file is actually in this folder, or pass the correct path."
        )

    is_xlsx = source_path.lower().endswith(".xlsx")
    if is_xlsx:
        print(f"Loading {source_path} ... (.xlsx files are slow to parse, this can take "
              f"1-3+ minutes for large files, this is NORMAL, please wait)", flush=True)
    else:
        print(f"Loading {source_path} ...", flush=True)

    t0 = time.time()
    if source_path.lower().endswith(".csv"):
        raw = pd.read_csv(source_path)
    else:
        raw = pd.read_excel(source_path)
    print(f"Loaded in {time.time()-t0:.1f}s. Raw rows: {len(raw):,}", flush=True)
    print(f"Columns found: {list(raw.columns)}", flush=True)

    print("Cleaning data...", flush=True)
    df = clean_transactions(raw)
    print(f"Cleaned rows: {len(df):,}", flush=True)

    print("Fitting K-Means model...", flush=True)
    pipeline = CustomerSegmentationPipeline(n_clusters=4, random_state=42)
    rfm = pipeline.fit(df)
    print(f"Fitted on {len(rfm):,} customers.", flush=True)

    print("\nSegment sizes:", flush=True)
    print(rfm["Segment"].value_counts())

    print(f"\nSaving to {output_path} ...", flush=True)
    pipeline.save(output_path)

    if os.path.exists(output_path):
        size_kb = os.path.getsize(output_path) / 1024
        print(f"CONFIRMED: {output_path} exists ({size_kb:.0f} KB) in {os.getcwd()}", flush=True)
    else:
        print(f"WARNING: save() ran but {output_path} was not found on disk afterward!", flush=True)

    print("\n" + "=" * 50, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print("=" * 50, flush=True)
    print("You can now launch the dashboard with: streamlit run app.py", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python train_pipeline.py "Online Retail.csv"')
        sys.exit(1)

    try:
        main(sys.argv[1])
    except Exception:
        print("\n" + "=" * 50, flush=True)
        print("TRAINING FAILED -- full error below", flush=True)
        print("=" * 50, flush=True)
        traceback.print_exc()
        print("=" * 50, flush=True)
        sys.exit(1)