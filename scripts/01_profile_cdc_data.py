import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import json
from pathlib import Path

import pandas as pd

from src.data_cdc import CDCQuery, fetch_cdc_rows


def main():
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    q = CDCQuery(limit=4000)
    df = fetch_cdc_rows(q)

    print("\n=== STAGE 01: DATA PROFILE ===")
    print("Rows:", len(df))
    print("Columns:", df.shape[1])

    print("\n--- Column names ---")
    for c in sorted(df.columns):
        print(c)

    print("\n--- dtypes ---")
    print(df.dtypes.sort_index())

    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    print("\n--- Missingness (% of rows) ---")
    print(missing)

    key_cols = [
        "yearstart",
        "locationabbr",
        "locationdesc",
        "question",
        "data_value",
        "low_confidence_limit",
        "high_confidence_limit",
        "sample_size",
        "age_years",
        "sex",
        "race_ethnicity",
        "education",
        "income",
    ]
    present_key_cols = [c for c in key_cols if c in df.columns]

    print("\n--- Sample values (first 3 non-null) ---")
    sample_values = {}
    for c in present_key_cols:
        vals = df[c].dropna().head(3).tolist()
        sample_values[c] = vals
        print(f"{c}: {vals}")

    out_csv = Path("data") / "cdc_sample_raw.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved raw sample to: {out_csv}")

    out_json = Path("reports") / "stage01_profile_summary.json"
    summary = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "missing_percent": missing.round(3).to_dict(),
        "sample_values": sample_values,
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved profile summary to: {out_json}")


if __name__ == "__main__":
    main()
