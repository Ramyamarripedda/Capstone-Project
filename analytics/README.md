# Titanic analytics

Run `01_eda.ipynb` followed by `02_modeling.ipynb`, or run `python eda.py`
then `python modeling.py`. Only `eda.py` calls the Seaborn raw loader, and
only when `titanic.csv` is absent. Later runs use the committed raw fallback.
Modeling continues from `data/titanic_clean.csv` produced by that EDA run.

`outputs/eda_interpretation.md` gives all missing percentages, age/fare IQR
outlier counts, fare skewness, survival breakdowns, the two strongest
correlations, four chart interpretations and an EDA z-score check. The EDA
copy fills age for description, but the shared modeling CSV leaves it missing
so the training-only pipeline learns the median.

`modeling.py` uses a stratified split and trains Logistic Regression, Decision
Tree and Random Forest with the same rows. It saves confusion matrices,
ROC curves, a labeled tree, a baseline/class-weight/SMOTE comparison, Random
Forest grid-search and OOB results, and a fare regression residual plot.
`outputs/model_interpretation.md` reports all metrics in separate classifier
and regression groups and recommends a classifier using actual values.
`models/best_classifier_pipeline.joblib` contains both the fitted preprocessing
and the selected estimator; reloading it gave the same predictions for five
raw held-out inputs. Load joblib artifacts only from trusted sources.
