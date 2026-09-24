# Titanic model results

## How the data was split

There are 889 cleaned rows: 549
non-survivors and 340 survivors. The split uses
711 rows for training and 178 rows for testing.
Stratification keeps a similar share of each class in both groups.
Each model fills missing values, encodes categories and scales numbers using
only its training rows. The test rows do not change these settings.
The `alive` column is left out because it directly reveals `survived`.

## Model comparison

| Model                  | Task           | Accuracy   | Precision   | Recall   | F1    | AUC   | MAE    | RMSE   | R2    | Adjusted R2   |
|:-----------------------|:---------------|:-----------|:------------|:---------|:------|:------|:-------|:-------|:------|:--------------|
| Logistic Regression    | Classification | 0.809      | 0.783       | 0.691    | 0.734 | 0.861 | -      | -      | -     | -             |
| Decision Tree          | Classification | 0.809      | 0.815       | 0.647    | 0.721 | 0.856 | -      | -      | -     | -             |
| Random Forest          | Classification | 0.815      | 0.778       | 0.721    | 0.748 | 0.818 | -      | -      | -     | -             |
| Fare Linear Regression | Regression     | -          | -           | -        | -     | -     | 21.100 | 41.702 | 0.348 | 0.317         |

Accuracy, precision, recall, F1 and AUC belong to the classification task.
MAE, RMSE, R2 and adjusted R2 belong to fare prediction. A dash means the
metric does not apply. Scores from these two tasks should not be compared.
Precision asks how many predicted survivors were correct. Recall asks how
many actual survivors were found. F1 balances these two values.

## Confusion matrices as numbers

| Model               |   TN |   FP |   FN |   TP |
|:--------------------|-----:|-----:|-----:|-----:|
| Logistic Regression |   97 |   13 |   21 |   47 |
| Decision Tree       |  100 |   10 |   24 |   44 |
| Random Forest       |   96 |   14 |   19 |   49 |

TN means correctly predicted non-survivors. FP means non-survivors predicted
as survivors. FN means survivors missed by the model. TP means correctly
predicted survivors. The PNG charts show these same counts. ROC curves are
also saved as plots and CSV point values, so the results can be checked in text.

## Model choice using training data

|                     |   Training CV F1 |
|:--------------------|-----------------:|
| Logistic Regression |           0.71   |
| Decision Tree       |           0.701  |
| Random Forest       |           0.7398 |
| Tuned Random Forest |           0.7675 |

These F1 scores come from five validation folds inside the training data.
The model with the highest training CV F1 is selected. Test-set F1 is not
used to choose the saved model. The tuned model's CV score was also used
to choose its settings, so the separate test score is the final check.

## Class imbalance

|                     |   precision |   recall |     f1 |
|:--------------------|------------:|---------:|-------:|
| baseline            |      0.7833 |   0.6912 | 0.7344 |
| balanced_weights    |      0.7183 |   0.75   | 0.7338 |
| smote_training_only |      0.7353 |   0.7353 | 0.7353 |

The best F1 in this comparison was smote_training_only
(0.7353). Differences this small on one
test split are not strong evidence of a clear winner. Balanced weights give
more importance to the smaller class. SMOTE creates extra training examples;
it never changes test rows. Ordinary SMOTE is used here as requested, although
its interpolated one-hot values may not represent real category combinations.

## Random Forest tuning

GridSearchCV tries n_estimators, max_depth and max_features using training data.
Best settings: `{'model__max_depth': 10, 'model__max_features': 0.8, 'model__n_estimators': 100}`.
The training CV F1 is 0.767.
The OOB accuracy is 0.831; it checks each
training row using trees that did not use that row. OOB accuracy and F1 are
different metrics. The tuned forest's test F1 is
0.788.

## Predicting fare

MAE is 21.100, RMSE is 41.702,
R2 is 0.348 and adjusted R2 is
0.317. MAE is the average size of an error in fare
units. RMSE gives more weight to large errors. Adjusted R2 uses
178 test rows and 8
predictor columns; one dummy level per category is dropped to avoid duplicate
information with the intercept. This is a descriptive test-set calculation.

The residual plot shows wider errors at higher predicted fares and some
large positive errors. This suggests heteroscedasticity: the error spread
is not constant. A simple linear model therefore misses part of the fare pattern.

## Recommendation

The selected classifier is **Tuned Random Forest**, based on training CV F1.
On the separate test set it has accuracy 0.843,
precision 0.812, recall 0.765,
F1 0.788 and AUC 0.827.
This makes it the choice for this class project, though the small historical
dataset does not prove it will work well on new data.
The saved file contains the fitted preprocessing and model together, and
reloading it gave the same predictions on five raw input rows:
**True**.
