from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests


@dataclass(frozen=True)
class CDCQuery:
    """
    Query the CDC Socrata API for rows from a dataset.

    Dataset used (Phase 1):
    - CDC Nutrition, Physical Activity, and Obesity (BRFSS-derived)
    Endpoint: https://data.cdc.gov/resource/hn4x-zwk7.json
    """
    base_url: str = "https://data.cdc.gov/resource/hn4x-zwk7.json"
    limit: int = 5000
    where: Optional[str] = None  


def fetch_cdc_rows(query: CDCQuery) -> pd.DataFrame:
    params = {"$limit": query.limit}
    if query.where:
        params["$where"] = query.where

    r = requests.get(query.base_url, params=params, timeout=30)
    r.raise_for_status()

    rows = r.json()
    df = pd.DataFrame(rows)

    return df


if __name__ == "__main__":
    # Small smoke test
    q = CDCQuery(limit=10)
    df = fetch_cdc_rows(q)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print(df.head(3).to_string(index=False))
