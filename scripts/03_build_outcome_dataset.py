import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_cdc import CDCQuery, fetch_cdc_rows


def _find_col(cols, required_substrings):
    required = [s.lower() for s in required_substrings]
    for c in cols:
        cl = str(c).lower()
        if all(s in cl for s in required):
            return c
    return None


def main():
    Path("data").mkdir(exist_ok=True)

    where = (
        "data_value IS NOT NULL "
        "AND yearstart IS NOT NULL "
        "AND locationabbr IS NOT NULL "
        "AND stratificationcategory1 IS NOT NULL "
        "AND stratification1 IS NOT NULL "
        "AND ("
        "lower(question) like '%obesity%' OR "
        "lower(question) like '%overweight%'"
        ")"
    )

    q = CDCQuery(limit=100000, where=where)
    df = fetch_cdc_rows(q)

    df = df[
        [
            "yearstart",
            "locationabbr",
            "stratificationcategory1",
            "stratification1",
            "question",
            "data_value",
        ]
    ].copy()

    df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")
    df["yearstart"] = pd.to_numeric(df["yearstart"], errors="coerce")

    df = df.dropna(
        subset=[
            "data_value",
            "yearstart",
            "locationabbr",
            "stratificationcategory1",
            "stratification1",
            "question",
        ]
    )
    df["yearstart"] = df["yearstart"].astype(int)

    wide = df.pivot_table(
        index=[
            "yearstart",
            "locationabbr",
            "stratificationcategory1",
            "stratification1",
        ],
        columns="question",
        values="data_value",
        aggfunc="mean",
    ).reset_index()

    base_cols = {
        "yearstart",
        "locationabbr",
        "stratificationcategory1",
        "stratification1",
    }
    question_cols = [c for c in wide.columns if c not in base_cols]

    obesity_col = _find_col(question_cols, ["percent", "adults", "18", "obesity"])
    overweight_col = _find_col(question_cols, ["percent", "adults", "18", "overweight"])

    if obesity_col is None or overweight_col is None:
        print("\nERROR: Could not identify obesity/overweight columns after pivot.")
        print("Available question columns:")
        for c in sorted(question_cols):
            print("-", c)
        raise SystemExit(
            "\nPaste the list above here and I will adjust the matching rules."
        )

    wide = wide.rename(
        columns={
            obesity_col: "obesity_value",
            overweight_col: "overweight_value",
        }
    )

    wide = wide.dropna(subset=["obesity_value", "overweight_value"])

    obesity_median = float(wide["obesity_value"].median())
    overweight_median = float(wide["overweight_value"].median())

    wide["obesity_high_risk"] = (wide["obesity_value"] >= obesity_median).astype(int)
    wide["overweight_high_risk"] = (wide["overweight_value"] >= overweight_median).astype(int)

    out_path = Path("data") / "obesity_overweight_modeling.csv"
    wide.to_csv(out_path, index=False)

    print("\n=== STAGE 03: BUILD OUTCOME DATASET ===")
    print("Rows:", len(wide))
    print("Obesity column used:", obesity_col)
    print("Overweight column used:", overweight_col)
    print("Obesity median threshold:", round(obesity_median, 3))
    print("Overweight median threshold:", round(overweight_median, 3))
    print("Obesity high-risk rate:", round(wide["obesity_high_risk"].mean(), 3))
    print("Overweight high-risk rate:", round(wide["overweight_high_risk"].mean(), 3))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
