import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_cdc import CDCQuery, fetch_cdc_rows


def main():
    q = CDCQuery(limit=70000)
    df = fetch_cdc_rows(q)

    counts = df["question"].value_counts(dropna=True).head(30)

    print("\n=== STAGE 02A: TOP QUESTIONS (by row count) ===")
    for question, n in counts.items():
        print(f"{n:6d}  {question}")


if __name__ == "__main__":
    main()
