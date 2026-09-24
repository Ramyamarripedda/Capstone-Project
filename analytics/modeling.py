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
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

ROOT = Path(__file__).resolve().parent
BASE_CSV = ROOT / "data" / "titanic_clean.csv"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
NUMERIC = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL = ["sex", "embarked"]


def preprocessor(numeric: list[str] = NUMERIC, drop_first: bool = False) -> ColumnTransformer:
    numeric_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                  drop="first" if drop_first else None)),
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
    confusion_counts = {}
    training_cv_f1 = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, estimator in estimators.items():
        model = Pipeline([("preprocess", preprocessor()), ("model", estimator)])
        training_cv_f1[name] = float(cross_val_score(
            model, x_train, y_train, cv=cv, scoring="f1", n_jobs=1).mean())
        model.fit(x_train, y_train)
        fitted[name] = model
        metrics[name] = classifier_metrics(model, x_test, y_test)
        confusion_counts[name] = confusion_matrix(
            y_test, model.predict(x_test), labels=[0, 1]).tolist()
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
        pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr,
                      "threshold": thresholds}).to_csv(
            OUT / f"roc_{name.lower().replace(' ', '_')}.csv", index=False)
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
        scoring="f1", cv=cv, n_jobs=1, refit=True,
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
        # Drop one level per category to avoid duplicate dummy columns with
        # the regression intercept and to count predictors for adjusted R2.
        ("preprocess", preprocessor(reg_numeric, drop_first=True)),
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
    training_cv_f1["Tuned Random Forest"] = grid_result["cross_validated_f1"]
    best_name = max(training_cv_f1, key=training_cv_f1.get)
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
        "confusion_matrices": confusion_counts, "training_cv_f1": training_cv_f1,
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
    selected = result["selected_metrics"]
    comparison_rows = []
    for name, values in result["classifiers"].items():
        comparison_rows.append([name, "Classification", *[f"{values[key]:.3f}" for key in
                                ("accuracy", "precision", "recall", "f1", "auc")],
                                "-", "-", "-", "-"])
    comparison_rows.append(["Fare Linear Regression", "Regression", "-", "-", "-", "-", "-",
                            *[f"{regression[key]:.3f}" for key in
                              ("mae", "rmse", "r2", "adjusted_r2")]])
    comparison = pd.DataFrame(comparison_rows, columns=[
        "Model", "Task", "Accuracy", "Precision", "Recall", "F1", "AUC",
        "MAE", "RMSE", "R2", "Adjusted R2"])
    matrices = []
    for name, matrix in result["confusion_matrices"].items():
        tn, fp = matrix[0]
        fn, tp = matrix[1]
        matrices.append([name, tn, fp, fn, tp])
    counts = pd.DataFrame(matrices, columns=["Model", "TN", "FP", "FN", "TP"])
    cv_table = pd.DataFrame.from_dict(result["training_cv_f1"], orient="index", columns=["Training CV F1"])
    best_imbalance = max(result["imbalance"], key=lambda name: result["imbalance"][name]["f1"])
    text = f"""# Titanic model results

## How the data was split

There are {sum(result['class_balance'].values())} cleaned rows: {result['class_balance']['0']}
non-survivors and {result['class_balance']['1']} survivors. The split uses
{result['train_rows']} rows for training and {result['test_rows']} rows for testing.
Stratification keeps a similar share of each class in both groups.
Each model fills missing values, encodes categories and scales numbers using
only its training rows. The test rows do not change these settings.
The `alive` column is left out because it directly reveals `survived`.

## Model comparison

{comparison.to_markdown(index=False)}

Accuracy, precision, recall, F1 and AUC belong to the classification task.
MAE, RMSE, R2 and adjusted R2 belong to fare prediction. A dash means the
metric does not apply. Scores from these two tasks should not be compared.
Precision asks how many predicted survivors were correct. Recall asks how
many actual survivors were found. F1 balances these two values.

## Confusion matrices as numbers

{counts.to_markdown(index=False)}

TN means correctly predicted non-survivors. FP means non-survivors predicted
as survivors. FN means survivors missed by the model. TP means correctly
predicted survivors. The PNG charts show these same counts. ROC curves are
also saved as plots and CSV point values, so the results can be checked in text.

## Model choice using training data

{cv_table.round(4).to_markdown()}

These F1 scores come from five validation folds inside the training data.
The model with the highest training CV F1 is selected. Test-set F1 is not
used to choose the saved model. The tuned model's CV score was also used
to choose its settings, so the separate test score is the final check.

## Class imbalance

{imbalance.round(4).to_markdown()}

The best F1 in this comparison was {best_imbalance}
({result['imbalance'][best_imbalance]['f1']:.4f}). Differences this small on one
test split are not strong evidence of a clear winner. Balanced weights give
more importance to the smaller class. SMOTE creates extra training examples;
it never changes test rows. Ordinary SMOTE is used here as requested, although
its interpolated one-hot values may not represent real category combinations.

## Random Forest tuning

GridSearchCV tries n_estimators, max_depth and max_features using training data.
Best settings: `{result['grid_search']['best_parameters']}`.
The training CV F1 is {result['grid_search']['cross_validated_f1']:.3f}.
The OOB accuracy is {result['grid_search']['oob_score']:.3f}; it checks each
training row using trees that did not use that row. OOB accuracy and F1 are
different metrics. The tuned forest's test F1 is
{result['grid_search']['test_metrics']['f1']:.3f}.

## Predicting fare

MAE is {regression['mae']:.3f}, RMSE is {regression['rmse']:.3f},
R2 is {regression['r2']:.3f} and adjusted R2 is
{regression['adjusted_r2']:.3f}. MAE is the average size of an error in fare
units. RMSE gives more weight to large errors. Adjusted R2 uses
{regression['test_rows']} test rows and {regression['transformed_features']}
predictor columns; one dummy level per category is dropped to avoid duplicate
information with the intercept. This is a descriptive test-set calculation.

The residual plot shows wider errors at higher predicted fares and some
large positive errors. This suggests heteroscedasticity: the error spread
is not constant. A simple linear model therefore misses part of the fare pattern.

## Recommendation

The selected classifier is **{best}**, based on training CV F1.
On the separate test set it has accuracy {selected['accuracy']:.3f},
precision {selected['precision']:.3f}, recall {selected['recall']:.3f},
F1 {selected['f1']:.3f} and AUC {selected['auc']:.3f}.
This makes it the choice for this class project, though the small historical
dataset does not prove it will work well on new data.
The saved file contains the fitted preprocessing and model together, and
reloading it gave the same predictions on five raw input rows:
**{result['saved_pipeline_reload_matches']}**.
"""
    (OUT / "model_interpretation.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    report = run()
    print("Classifiers:", report["classifiers"])
    print("Best:", report["best_classifier"], "reloaded:", report["saved_pipeline_reload_matches"])
