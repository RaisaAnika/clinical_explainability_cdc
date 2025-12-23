import pandas as pd

from data_cdc import CDCQuery, fetch_cdc_rows


# def main():
#     q = CDCQuery(limit=2000)
#     df = fetch_cdc_rows(q)

#     print("Total rows:", len(df))
#     print("\nColumns:")
#     for c in sorted(df.columns):
#         print(" -", c)

#     print("\nNumeric-like columns (sample):")
#     numeric_candidates = [
#         "data_value",
#         "low_confidence_limit",
#         "high_confidence_limit",
#         "sample_size",
#         "yearstart",
#     ]

#     for col in numeric_candidates:
#         if col in df.columns:
#             print(f"\n{col}:")
#             print(df[col].dropna().head(5))


# if __name__ == "__main__":
#     main()




def main():
    q = CDCQuery(limit=2000)
    df = fetch_cdc_rows(q)

    print("Columns:")
    for col in sorted(df.columns):
        print(col)

    print ("\n Number of Columns ", len(df))


if __name__ == "__main__":
    main()