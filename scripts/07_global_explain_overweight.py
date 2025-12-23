import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def main():
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)

    pipe = joblib.load("models/logreg_overweight.joblib")

    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    cat_encoder = pre.named_transformers_["cat"]
    cat_features = cat_encoder.get_feature_names_out(
        ["locationabbr", "stratificationcategory1", "stratification1"]
    )

    feature_names = list(cat_features) + ["yearstart"]
    coefs = model.coef_[0]

    imp = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefs,
            }
        )
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
    )

    imp.to_csv("reports/tables/overweight_global_importance.csv", index=False)

    top = imp.head(20)

    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["coefficient"])
    plt.gca().invert_yaxis()
    plt.xlabel("Logistic Regression Coefficient")
    plt.title("Top Global Drivers — Overweight Risk")
    plt.tight_layout()
    plt.savefig("reports/figures/global_importance_overweight.png")
    plt.close()

    print("\n=== GLOBAL EXPLAINABILITY: OVERWEIGHT ===")
    print(top.head(10).to_string(index=False))
    print("\nSaved:")
    print(" - reports/tables/overweight_global_importance.csv")
    print(" - reports/figures/global_importance_overweight.png")


if __name__ == "__main__":
    main()
