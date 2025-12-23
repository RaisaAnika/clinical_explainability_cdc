import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def train_for_label(df: pd.DataFrame, label_col: str, model_out: str):
    feature_cols = [
        "yearstart",
        "locationabbr",
        "stratificationcategory1",
        "stratification1",
    ]

    X = df[feature_cols].copy()
    y = df[label_col].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical = ["locationabbr", "stratificationcategory1", "stratification1"]
    numeric = ["yearstart"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    clf = LogisticRegression(max_iter=5000)

    pipe = Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", clf),
        ]
    )

    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    Path("models").mkdir(exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, Path("models") / model_out)

    print(f"\n=== TRAIN RESULT: {label_col} ===")
    print("Test AUROC:", round(auc, 4))
    print("Test Accuracy:", round(acc, 4))
    print("Confusion Matrix [ [TN FP] [FN TP] ]:")
    print(cm)
    print("Saved model:", f"models/{model_out}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {label_col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"reports/figures/roc_{label_col}.png")
    plt.close()

    # Confusion matrix plot
    plt.figure()
    plt.imshow(cm)
    plt.title(f"Confusion Matrix — {label_col}")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(f"reports/figures/cm_{label_col}.png")
    plt.close()


def main():
    df = pd.read_csv("data/obesity_overweight_modeling.csv")
    train_for_label(df, "obesity_high_risk", "logreg_obesity.joblib")


if __name__ == "__main__":
    main()
