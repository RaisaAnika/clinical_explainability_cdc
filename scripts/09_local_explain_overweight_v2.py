import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    pipe = joblib.load("models/logreg_overweight_v2.joblib")
    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    df = pd.read_csv("data/obesity_overweight_modeling.csv")

    example = df.sample(1, random_state=7)

    X = example[
        [
            "yearstart",
            "locationabbr",
            "stratificationcategory1",
            "stratification1",
        ]
    ]

    Xt = pre.transform(X)

    cat_encoder = pre.named_transformers_["cat"]
    cat_features = cat_encoder.get_feature_names_out(
        ["locationabbr", "stratificationcategory1", "stratification1"]
    )

    feature_names = list(cat_features) + ["yearstart_scaled"]

    coefs = model.coef_[0]
    intercept = model.intercept_[0]

    vec = Xt.toarray()[0]
    contributions = vec * coefs

    contrib_df = (
        pd.DataFrame({"feature": feature_names, "contribution": contributions})
        .assign(abs_contribution=lambda d: d["contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
    )

    log_odds = intercept + contributions.sum()
    prob = sigmoid(log_odds)

    print("\n=== LOCAL EXPLANATION (V2): OVERWEIGHT ===")
    print("Example subgroup:")
    print(
        example[
            [
                "yearstart",
                "locationabbr",
                "stratificationcategory1",
                "stratification1",
                "overweight_value",
                "overweight_high_risk",
            ]
        ].to_string(index=False)
    )

    print("\nTop contributing features:")
    print(contrib_df.head(10).to_string(index=False))

    print("\nBase intercept (log-odds):", round(intercept, 4))
    print("Sum of contributions:", round(contributions.sum(), 4))
    print("Final predicted probability:", round(prob, 4))


if __name__ == "__main__":
    main()
