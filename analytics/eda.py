"""Load Titanic once, profile it, clean for EDA, and save real charts/results."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "titanic.csv"
BASE_CSV = ROOT / "data" / "titanic_clean.csv"
OUT = ROOT / "outputs"
CORRELATION_COLUMNS = ["survived", "pclass", "age", "sibsp", "parch", "fare"]


def save_plot(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=130)
    plt.close()


def load_raw() -> pd.DataFrame:
    """Exactly one Seaborn call in the project, used only when fallback is absent."""
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)
    raw = sns.load_dataset("titanic")
    raw.to_csv(RAW_CSV, index=False)
    return raw


def make_clean_base(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = raw.isna().mean().mul(100)
    affected = {name: round(float(rate), 3) for name, rate in missing.items() if rate > 0}
    # cabin has too little observed data; embark_town repeats embarked.
    cleaned = raw.drop(columns=["deck", "embark_town"])
    # Both are under 5% missing in the supplied dataset.
    cleaned = cleaned.dropna(subset=["embarked"])
    # Keep age NaN in the shared file. EDA uses a median-filled copy, while the
    # predictive pipeline learns an age median from training rows only.
    return cleaned.reset_index(drop=True), affected


def outlier_count(series: pd.Series) -> tuple[int, float, float]:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < low) | (series > high)).sum()), float(low), float(high)


def masked_rate(frame: pd.DataFrame, mask: pd.Series) -> float:
    return float(frame.loc[mask, "survived"].mean())


def run() -> dict:
    OUT.mkdir(exist_ok=True)
    BASE_CSV.parent.mkdir(exist_ok=True)
    raw = load_raw()
    base, missing = make_clean_base(raw)
    base.to_csv(BASE_CSV, index=False)
    eda = base.copy()
    eda["age"] = eda["age"].fillna(eda["age"].median())

    # Requested profiling is on the loaded raw data, before modifications.
    profile_path = OUT / "profile.txt"
    with profile_path.open("w", encoding="utf-8") as stream:
        raw.info(buf=stream)
        stream.write("\nShape: " + str(raw.shape) + "\n\n")
        stream.write(raw.describe().to_string() + "\n\n")
        stream.write("Missing percentages:\n" + pd.Series(missing).to_string() + "\n")
    profile_path.write_text(
        "\n".join(line.rstrip() for line in profile_path.read_text(encoding="utf-8").splitlines())
        + "\n", encoding="utf-8")

    outliers = {}
    fare_stats = {"mean": float(eda.fare.mean()), "median": float(eda.fare.median()),
                  "mode": float(eda.fare.mode().iloc[0])}
    for col in ("age", "fare"):
        count, low, high = outlier_count(eda[col])
        outliers[col] = {"count": count, "lower": low, "upper": high}
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(eda[col], ax=axes[0], bins=25)
        sns.boxplot(y=eda[col], ax=axes[1])
        axes[0].set_title(f"{col.capitalize()} histogram")
        axes[1].set_title(f"{col.capitalize()} box plot")
        save_plot(f"{col}_hist_box.png")

    # Boolean masking explicitly uses & for the joint breakdown.
    sex_rates = {sex: masked_rate(eda, eda.sex.eq(sex)) for sex in sorted(eda.sex.unique())}
    class_rates = {str(pclass): masked_rate(eda, eda.pclass.eq(pclass))
                   for pclass in sorted(eda.pclass.unique())}
    joint_rates = {f"{sex}, class {pclass}": masked_rate(
        eda, (eda.sex.eq(sex)) & (eda.pclass.eq(pclass)))
        for sex in sorted(eda.sex.unique()) for pclass in sorted(eda.pclass.unique())}
    women_or_first_class = masked_rate(
        eda, eda.sex.eq("female") | eda.pclass.eq(1))

    corr = eda[CORRELATION_COLUMNS].corr()
    corr.to_csv(OUT / "correlations.csv")
    pairs = sorted(((left, right, float(corr.loc[left, right]))
                    for left, right in combinations(CORRELATION_COLUMNS, 2)),
                   key=lambda item: abs(item[2]), reverse=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Six measured Titanic variables")
    save_plot("correlation_heatmap.png")

    # Four distinct multivariate views. The written interpretation below uses
    # numbers calculated here, so it cannot silently disagree with the plots.
    plt.figure(figsize=(7, 4))
    sns.barplot(data=eda, x="pclass", y="survived", hue="sex", errorbar=None)
    plt.ylabel("Survival rate")
    plt.title("Survival by class and sex")
    save_plot("story_1_sex_class.png")

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=eda, x="pclass", y="age", hue="survived")
    plt.title("Age by class and survival")
    save_plot("story_2_age_class_survival.png")

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=eda, x="sex", y="fare", hue="survived", showfliers=False)
    plt.title("Fare by sex and survival (outliers hidden in view)")
    save_plot("story_3_fare_sex_survival.png")

    eda["family_size"] = eda.sibsp + eda.parch + 1
    eda["family_group"] = pd.cut(eda.family_size, bins=[0, 1, 4, np.inf],
                                 labels=["alone", "2-4", "5+"])
    family_rates = eda.groupby(["family_group", "pclass"], observed=True).survived.mean()
    plt.figure(figsize=(8, 4))
    sns.barplot(data=eda, x="family_group", y="survived", hue="pclass", errorbar=None)
    plt.ylabel("Survival rate")
    plt.title("Family size, class and survival")
    save_plot("story_4_family_class_survival.png")

    scaling = {}
    for col in ("age", "fare"):
        before_mean = float(eda[col].mean())
        before_std = float(eda[col].std(ddof=0))
        z = (eda[col] - before_mean) / before_std
        scaling[col] = {"before_mean": before_mean, "before_std": before_std,
                        "after_mean": float(z.mean()), "after_std": float(z.std(ddof=0))}

    findings = {
        "raw_shape": list(raw.shape), "clean_shape": list(base.shape),
        "missing_percent": missing, "age_median_for_eda": float(base.age.median()),
        "outliers": outliers, "fare": fare_stats, "sex_rates": sex_rates,
        "class_rates": class_rates, "sex_class_rates": joint_rates,
        "women_or_first_class_rate": women_or_first_class,
        "top_correlations": pairs[:2], "scaling": scaling,
        "family_class_rates": {f"{group}, class {pclass}": float(value)
                               for (group, pclass), value in family_rates.items()},
        "class_balance": {str(k): float(v) for k, v in base.survived.value_counts(normalize=True).items()},
    }
    (OUT / "eda_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    write_interpretation(findings, eda)
    return findings


def write_interpretation(f: dict, eda: pd.DataFrame) -> None:
    sex = f["sex_rates"]
    cls = f["class_rates"]
    fare = f["fare"]
    female_first = f["sex_class_rates"]["female, class 1"]
    male_third = f["sex_class_rates"]["male, class 3"]
    age_by_survival = eda.groupby("survived").age.median().to_dict()
    fare_by_survival = eda.groupby("survived").fare.median().to_dict()
    pair_explanations = []
    for left, right, value in f["top_correlations"]:
        if {left, right} == {"pclass", "fare"}:
            explanation = ("The negative value means lower numeric class "
                           "codes (higher travel classes) tend to have higher fares.")
        elif {left, right} == {"sibsp", "parch"}:
            explanation = ("The positive value means passengers traveling "
                           "with more siblings/spouses also tend to have more "
                           "parents/children on board.")
        else:
            explanation = ("The variables tend to move together." if value > 0
                           else "The variables tend to move in opposite directions.")
        pair_explanations.append(explanation)
    text = f"""# Titanic EDA results

Raw data: {f['raw_shape'][0]} rows and {f['raw_shape'][1]} columns. After removing
rows with missing embarked, the shared clean base has {f['clean_shape'][0]} rows.
The original raw dataset is saved in `analytics/titanic.csv`.

## Missing values and decisions

{chr(10).join(f'- {name}: {rate:.3f}% missing.' for name, rate in f['missing_percent'].items())}

The under-5% missing `embarked` rows were dropped. `embark_town` has the same
under-5% pattern but repeats the port in `embarked`, so it was removed as a
redundant column rather than dropping a second set of rows. `age` is in the
5%-30% band: the EDA view uses its median ({f['age_median_for_eda']:.2f}) for
plots. The shared modeling input retains missing ages so its median is learned
only from the training split. `deck` is more than 30% missing and was dropped
because a mostly empty column is unreliable for imputation.

## One-variable analysis

IQR outliers: age {f['outliers']['age']['count']} outside
[{f['outliers']['age']['lower']:.2f}, {f['outliers']['age']['upper']:.2f}],
fare {f['outliers']['fare']['count']} outside
[{f['outliers']['fare']['lower']:.2f}, {f['outliers']['fare']['upper']:.2f}].
Fare has mean {fare['mean']:.2f}, median {fare['median']:.2f}, and mode
{fare['mode']:.2f}. Since mean > median > mode, it is right-skewed; the high
fare tail in the histogram and box plot supports this description.

## Two-variable analysis

Survival by sex: {', '.join(f'{name} {value:.1%}' for name, value in sex.items())}.
Survival by class: {', '.join(f'class {name} {value:.1%}' for name, value in cls.items())}.
Survival by sex and class: {', '.join(f'{name} {value:.1%}' for name, value in f['sex_class_rates'].items())}.
These rates were calculated with Boolean masks, including `&` for joint groups.
An additional `|` mask for women **or** first-class passengers gives
{f['women_or_first_class_rate']:.1%} survival; its groups overlap and therefore
should not be added together.

The heatmap uses exactly survived, pclass, age, sibsp, parch and fare. The two
largest absolute off-diagonal correlations are
{f['top_correlations'][0][0]} vs {f['top_correlations'][0][1]}
({f['top_correlations'][0][2]:+.3f}) and
{f['top_correlations'][1][0]} vs {f['top_correlations'][1][1]}
({f['top_correlations'][1][2]:+.3f}). The sign shows whether each pair tends
to move together or in opposite directions. {pair_explanations[0]}
{pair_explanations[1]} Correlation alone does not prove a cause of survival.

## Four-chart survival story

1. **Sex and class:** Survival is {female_first:.1%} for first-class women and
{male_third:.1%} for third-class men. The gap shows that both recorded sex and
passenger class are useful signals. It does not isolate either one's causal effect.

2. **Age, class and survival:** Median age is {age_by_survival.get(1, float('nan')):.1f}
among survivors and {age_by_survival.get(0, float('nan')):.1f} among non-survivors
in the EDA view. The overlapping boxes show that age alone does not cleanly
separate outcomes. Class changes the age mix within both groups.

3. **Fare, sex and survival:** Median fare is {fare_by_survival.get(1, float('nan')):.2f}
for survivors and {fare_by_survival.get(0, float('nan')):.2f} for non-survivors.
The plot hides individual extreme points so the central boxes can be compared;
the separate fare box plot still counts the outliers. Fare likely reflects class
as well as other booking details.

4. **Family size and class:** The plot compares solo passengers, groups of 2-4,
and groups of 5+ within each class. The actual survival rates are
{', '.join(f'{name} {rate:.1%}' for name, rate in f['family_class_rates'].items())}.
Both family size and class vary across groups, so the chart is descriptive
rather than a causal explanation.

## EDA scaling check

{chr(10).join(f'- {name}: mean {values["before_mean"]:.4f} -> {values["after_mean"]:.6f}; population standard deviation {values["before_std"]:.4f} -> {values["after_std"]:.6f}.' for name, values in f['scaling'].items())}

These full-EDA z-scores are only a display check; modeling fits its own scaler
on the training split.
"""
    (OUT / "eda_interpretation.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    result = run()
    print("Raw/clean:", result["raw_shape"], result["clean_shape"])
    print("Missing:", result["missing_percent"])
    print("Top correlations:", result["top_correlations"])
