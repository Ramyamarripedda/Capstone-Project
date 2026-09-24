# Titanic analysis

This part studies which passengers survived and then builds prediction models.
Run `01_eda.ipynb` before `02_modeling.ipynb`. The matching scripts are `eda.py`
and `modeling.py`. Follow the setup in the root README.

The first EDA run loads Titanic through Seaborn and saves `titanic.csv`.
Later runs read that file. Both notebooks use this one dataset. Two rows
with missing port information are removed. Deck is mostly missing, and
embark_town repeats embarked, so these two columns are removed.

Age is filled with its median for EDA plots. The shared file
`data/titanic_clean.csv` keeps missing ages so each training pipeline learns
its own median after the split. This avoids using test information in training.
The `alive` column directly reveals the target and is not used as a predictor.

Read these result files:

- `outputs/eda_interpretation.md`: missing percentages, outliers, survival rates,
  strongest correlations, chart explanations and the z-score check.
- `outputs/model_interpretation.md`: three classifier scores, confusion counts,
  training CV scores, imbalance comparison, tuning, regression and model choice.
- `outputs/model_results.json`: the same measured results in a code-readable form.

There are three classifiers: Logistic Regression, Decision Tree and Random
Forest. The training data is used for preprocessing, cross-validation and
tuning. The test data is used to report performance. SMOTE changes training
rows only. The highest training CV F1 decides which complete pipeline is saved
in `models/best_classifier_pipeline.joblib`.

Fare prediction is a separate linear-regression task. Its errors use different
units from classification scores. One category level is dropped for regression
so redundant dummy columns are not counted in adjusted R2.

The saved pipeline is checked by reloading it and predicting from five raw
rows. Only load joblib files from a trusted source.
