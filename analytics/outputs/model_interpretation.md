# Titanic modeling results

The shared cleaned CSV contains 889 rows:
549 did not survive and 340
survived. A stratified 80/20 split (711 training,
178 test rows) keeps this unequal balance similar in both
sets. Imputation, encoding and standardization are fitted by each pipeline
only on training rows; the held-out test rows are transformed without fitting.
Age uses a training median; categories use a training most-frequent value.

## Model comparison: separate classifier and regression metric groups

| Model                  | Type       | Accuracy   | Precision   | Recall   | F1    | AUC   | MAE    | RMSE   | R2    | Adjusted R2   |
|:-----------------------|:-----------|:-----------|:------------|:---------|:------|:------|:-------|:-------|:------|:--------------|
| Logistic Regression    | Classifier | 0.809      | 0.783       | 0.691    | 0.734 | 0.861 | —      | —      | —     | —             |
| Decision Tree          | Classifier | 0.809      | 0.815       | 0.647    | 0.721 | 0.856 | —      | —      | —     | —             |
| Random Forest          | Classifier | 0.815      | 0.778       | 0.721    | 0.748 | 0.818 | —      | —      | —     | —             |
| Fare Linear Regression | Regression | —          | —           | —        | —     | —     | 21.100 | 41.702 | 0.348 | 0.309         |

Classification scores and fare errors use different scales and should not be
ranked against each other. The grouped table leaves unrelated metric columns
blank for each model type.

## Classification metrics

|                     |   accuracy |   precision |   recall |    f1 |   auc |
|:--------------------|-----------:|------------:|---------:|------:|------:|
| Logistic Regression |      0.809 |       0.783 |    0.691 | 0.734 | 0.861 |
| Decision Tree       |      0.809 |       0.815 |    0.647 | 0.721 | 0.856 |
| Random Forest       |      0.815 |       0.778 |    0.721 | 0.748 | 0.818 |

The three saved confusion matrix plots give raw prediction counts; the ROC plot
shows the threshold trade-off for the same held-out split. The labeled tree
plot shows its depth-four splits with transformed feature names.

## Imbalance comparison

|                     |   precision |   recall |    f1 |
|:--------------------|------------:|---------:|------:|
| baseline            |       0.783 |    0.691 | 0.734 |
| balanced_weights    |       0.718 |    0.75  | 0.734 |
| smote_training_only |       0.735 |    0.735 | 0.735 |

Among these three logistic-regression variants, **smote_training_only** had the
highest held-out F1 (0.735).
That is the best precision/recall balance under this chosen metric; compare the
other precision and recall values before choosing a strategy for a real service.
SMOTE ran only inside the training pipeline, never on held-out test rows.

## Random Forest tuning

Five-fold GridSearchCV searched n_estimators, max_depth and max_features on
training data. Best parameters: `{'model__max_depth': 10, 'model__max_features': 0.8, 'model__n_estimators': 200}`.
Cross-validated F1: 0.767.
The refitted forest's out-of-bag score: 0.833.
Its held-out F1 is 0.788.
The OOB value is a training-only estimate and should not be confused with
held-out F1 or accuracy.

## Fare regression metrics (a separate scale)

| MAE | RMSE | R2 | Adjusted R2 |
| ---: | ---: | ---: | ---: |
| 21.100 | 41.702 | 0.348 | 0.309 |

Adjusted R2 uses 178 held-out rows and
10 transformed predictor columns. Inspect
`fare_residuals.png`: the spread is narrow near low predicted fares but widens
at higher predicted fares, with several large positive residuals. This is
evidence of heteroscedasticity, so the linear model's constant-variance
assumption is questionable for these data.

## Classifier recommendation

By held-out F1, I would choose **Tuned Random Forest** after comparing the three required
models and the tuned forest. It
achieved F1 0.788, precision 0.812,
recall 0.765, and AUC 0.831 on the same
test split. These numbers describe performance on this historical dataset and
do not establish how it would work on new passengers. The saved artifact
contains both preprocessing and the fitted classifier; reloading it reproduced
predictions for five raw held-out rows: **True**.
