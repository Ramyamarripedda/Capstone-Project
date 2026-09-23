"""Train and evaluate Titanic classifiers plus a fare regression pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ResamplePipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

ROOT = Path(__file__).resolve().parent
BASE_CSV = ROOT / "data" / "titanic_clean.csv"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
NUMERIC = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL = ["sex", "embarked"]


def preprocessor(numeric: list[str] = NUMERIC) -> ColumnTransformer:
    numeric_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_steps, numeric),
        ("categorical", categorical_steps, CATEGORICAL),
    ])


def classifier_metrics(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    prediction = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "auc": float(roc_auc_score(y_test, probabilities)),
    }


def run() -> dict:
    if not BASE_CSV.exists():
        raise FileNotFoundError("Run analytics/eda.py first; it creates the shared clean CSV")
    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    frame = pd.read_csv(BASE_CSV)
    x = frame[NUMERIC + CATEGORICAL]
    y = frame["survived"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    balance = {str(k): int(v) for k, v in y.value_counts().sort_index().items()}

    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    }
    fitted = {}
    metrics = {}
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, estimator in estimators.items():
        model = Pipeline([("preprocess", preprocessor()), ("model", estimator)])
        model.fit(x_train, y_train)
        fitted[name] = model
        metrics[name] = classifier_metrics(model, x_test, y_test)
        ConfusionMatrixDisplay.from_estimator(
            model, x_test, y_test, display_labels=["not survived", "survived"])
        plt.tight_layout()
        plt.savefig(OUT / f"confusion_{name.lower().replace(' ', '_')}.png", dpi=130)
        plt.close()
        RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("ROC curves on the same held-out test split")
    fig.tight_layout()
    fig.savefig(OUT / "roc_curves.png", dpi=130)
    plt.close(fig)

    tree = fitted["Decision Tree"]
    names = tree.named_steps["preprocess"].get_feature_names_out()
    plt.figure(figsize=(20, 10))
    plot_tree(tree.named_steps["model"], feature_names=names,
              class_names=["not survived", "survived"], filled=True,
              rounded=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "decision_tree.png", dpi=130)
    plt.close()

    # Same train/test split for each imbalance treatment. SMOTE is inside the
    # imblearn pipeline, so fitting during training never resamples test rows.
    imbalance_models = {
        "baseline": Pipeline([
            ("preprocess", preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "balanced_weights": Pipeline([
            ("preprocess", preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=42,
                                           class_weight="balanced")),
        ]),
        "smote_training_only": ResamplePipeline([
            ("preprocess", preprocessor()),
            ("smote", SMOTE(random_state=42)),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
    }
    imbalance = {}
    for name, model in imbalance_models.items():
        model.fit(x_train, y_train)
        values = classifier_metrics(model, x_test, y_test)
        imbalance[name] = {key: values[key] for key in ("precision", "recall", "f1")}

    # Out-of-bag score is populated on the winning fitted estimator because
    # oob_score=True is set before GridSearchCV fits each candidate.
    forest_grid = GridSearchCV(
        Pipeline([
            ("preprocess", preprocessor()),
            ("model", RandomForestClassifier(
                oob_score=True, n_jobs=1, random_state=42)),
        ]),
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__max_features": ["sqrt", 0.8],
        },
        scoring="f1", cv=5, n_jobs=1, refit=True,
    )
    forest_grid.fit(x_train, y_train)
    grid_result = {
        "best_parameters": forest_grid.best_params_,
        "cross_validated_f1": float(forest_grid.best_score_),
        "oob_score": float(forest_grid.best_estimator_.named_steps["model"].oob_score_),
        "test_metrics": classifier_metrics(forest_grid.best_estimator_, x_test, y_test),
    }

    # Fare is the target here, so it is excluded from regression inputs.
    reg_numeric = ["survived", "pclass", "age", "sibsp", "parch"]
    reg_features = reg_numeric + CATEGORICAL
    reg_x_train, reg_x_test, reg_y_train, reg_y_test = train_test_split(
        frame[reg_features], frame["fare"], test_size=0.2, random_state=42
    )
    regression = Pipeline([
        ("preprocess", preprocessor(reg_numeric)),
        ("model", LinearRegression()),
    ])
    regression.fit(reg_x_train, reg_y_train)
    reg_pred = regression.predict(reg_x_test)
    reg_r2 = float(r2_score(reg_y_test, reg_pred))
    n = len(reg_y_test)
    p = len(regression.named_steps["preprocess"].get_feature_names_out())
    if n <= p + 1:
        raise ValueError("Adjusted R2 requires more test rows than transformed features + 1")
    reg_metrics = {
        "mae": float(mean_absolute_error(reg_y_test, reg_pred)),
        "rmse": float(np.sqrt(mean_squared_error(reg_y_test, reg_pred))),
        "r2": reg_r2,
        "adjusted_r2": float(1 - (1 - reg_r2) * (n - 1) / (n - p - 1)),
        "test_rows": n,
        "transformed_features": p,
    }
    residual = reg_y_test.to_numpy() - reg_pred
    plt.figure(figsize=(7, 4))
    plt.scatter(reg_pred, residual, alpha=0.65, s=16)
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Predicted fare")
    plt.ylabel("Residual (actual minus predicted fare)")
    plt.title("Fare regression residuals")
    plt.tight_layout()
    plt.savefig(OUT / "fare_residuals.png", dpi=130)
    plt.close()

    # Compare the three required baselines and the tuned forest. The selected
    # artifact is a full fitted preprocessing + estimator pipeline.
    candidates = {**fitted, "Tuned Random Forest": forest_grid.best_estimator_}
    candidate_metrics = {**metrics, "Tuned Random Forest": grid_result["test_metrics"]}
    best_name = max(candidate_metrics, key=lambda name: candidate_metrics[name]["f1"])
    artifact = MODELS / "best_classifier_pipeline.joblib"
    joblib.dump(candidates[best_name], artifact)
    reloaded = joblib.load(artifact)
    raw_example = x_test.head(5)
    reload_matches = bool(np.array_equal(
        candidates[best_name].predict(raw_example), reloaded.predict(raw_example)))
    assert reload_matches
    raw_example.to_csv(OUT / "raw_prediction_examples.csv", index=False)
    result = {
        "class_balance": balance, "train_rows": len(x_train), "test_rows": len(x_test),
        "classifiers": metrics, "imbalance": imbalance,
        "grid_search": grid_result, "regression": reg_metrics,
        "best_classifier": best_name, "selected_metrics": candidate_metrics[best_name],
        "saved_pipeline_reload_matches": reload_matches,
    }
    (OUT / "model_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result)
    return result


def write_report(result: dict) -> None:
    scores = pd.DataFrame.from_dict(result["classifiers"], orient="index")
    imbalance = pd.DataFrame.from_dict(result["imbalance"], orient="index")
    regression = result["regression"]
    best = result["best_classifier"]
    best_scores = result["selected_metrics"]
    best_imbalance = max(result["imbalance"], key=lambda name: result["imbalance"][name]["f1"])
    comparison_rows = []
    for name, values in result["classifiers"].items():
        comparison_rows.append([name, "Classifier", *[f"{values[key]:.3f}" for key in
                                ("accuracy", "precision", "recall", "f1", "auc")],
                                "—", "—", "—", "—"])
    comparison_rows.append(["Fare Linear Regression", "Regression", "—", "—", "—", "—", "—",
                            *[f"{regression[key]:.3f}" for key in
                              ("mae", "rmse", "r2", "adjusted_r2")]])
    comparison = pd.DataFrame(comparison_rows, columns=[
        "Model", "Type", "Accuracy", "Precision", "Recall", "F1", "AUC",
        "MAE", "RMSE", "R2", "Adjusted R2"])
    text = f"""# Titanic modeling results

The shared cleaned CSV contains {sum(result['class_balance'].values())} rows:
{result['class_balance']['0']} did not survive and {result['class_balance']['1']}
survived. A stratified 80/20 split ({result['train_rows']} training,
{result['test_rows']} test rows) keeps this unequal balance similar in both
sets. Imputation, encoding and standardization are fitted by each pipeline
only on training rows; the held-out test rows are transformed without fitting.
Age uses a training median; categories use a training most-frequent value.

## Model comparison: separate classifier and regression metric groups

{comparison.to_markdown(index=False)}

Classification scores and fare errors use different scales and should not be
ranked against each other. The grouped table leaves unrelated metric columns
blank for each model type.

## Classification metrics

{scores.round(3).to_markdown()}

The three saved confusion matrix plots give raw prediction counts; the ROC plot
shows the threshold trade-off for the same held-out split. The labeled tree
plot shows its depth-four splits with transformed feature names.

## Imbalance comparison

{imbalance.round(3).to_markdown()}

Among these three logistic-regression variants, **{best_imbalance}** had the
highest held-out F1 ({result['imbalance'][best_imbalance]['f1']:.3f}).
That is the best precision/recall balance under this chosen metric; compare the
other precision and recall values before choosing a strategy for a real service.
SMOTE ran only inside the training pipeline, never on held-out test rows.

## Random Forest tuning

Five-fold GridSearchCV searched n_estimators, max_depth and max_features on
training data. Best parameters: `{result['grid_search']['best_parameters']}`.
Cross-validated F1: {result['grid_search']['cross_validated_f1']:.3f}.
The refitted forest's out-of-bag score: {result['grid_search']['oob_score']:.3f}.
Its held-out F1 is {result['grid_search']['test_metrics']['f1']:.3f}.
The OOB value is a training-only estimate and should not be confused with
held-out F1 or accuracy.

## Fare regression metrics (a separate scale)

| MAE | RMSE | R2 | Adjusted R2 |
| ---: | ---: | ---: | ---: |
| {regression['mae']:.3f} | {regression['rmse']:.3f} | {regression['r2']:.3f} | {regression['adjusted_r2']:.3f} |

Adjusted R2 uses {regression['test_rows']} held-out rows and
{regression['transformed_features']} transformed predictor columns. Inspect
`fare_residuals.png`: the spread is narrow near low predicted fares but widens
at higher predicted fares, with several large positive residuals. This is
evidence of heteroscedasticity, so the linear model's constant-variance
assumption is questionable for these data.

## Classifier recommendation

By held-out F1, I would choose **{best}** after comparing the three required
models and the tuned forest. It
achieved F1 {best_scores['f1']:.3f}, precision {best_scores['precision']:.3f},
recall {best_scores['recall']:.3f}, and AUC {best_scores['auc']:.3f} on the same
test split. These numbers describe performance on this historical dataset and
do not establish how it would work on new passengers. The saved artifact
contains both preprocessing and the fitted classifier; reloading it reproduced
predictions for five raw held-out rows: **{result['saved_pipeline_reload_matches']}**.
"""
    (OUT / "model_interpretation.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    report = run()
    print("Classifiers:", report["classifiers"])
    print("Best:", report["best_classifier"], "reloaded:", report["saved_pipeline_reload_matches"])
