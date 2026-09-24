# Titanic EDA results

Raw data: 891 rows and 15 columns. After removing
rows with missing embarked, the shared clean base has 889 rows.
The original raw dataset is saved in `analytics/titanic.csv`.
It has 549 non-survivors and 342 survivors.
EDA means looking at the data before fitting models.

## Missing values and decisions

- age: 19.865% missing.
- embarked: 0.224% missing.
- deck: 77.217% missing.
- embark_town: 0.224% missing.

Both port columns, `embarked` and `embark_town`, have less than 5% missing
data. Rows missing either value were dropped; these were the same two rows.
The repeated `embark_town` column was then removed. `age` is in the
5%-30% band: the EDA view uses its median (28.00) for
plots. For model training, age is kept missing until the split. The model learns
the replacement median from training rows only. This prevents test data from
influencing training. `deck` is more than 30% missing and was dropped
because a mostly empty column is unreliable for imputation.

## Looking at age and fare

IQR outliers: age 65 outside
[2.50, 54.50],
fare 114 outside
[-26.76, 65.66].
Fare has mean 32.10, median 14.45, and mode
8.05. Since mean > median > mode, it is right-skewed; the high
fare tail in the histogram and box plot supports this description.

## Comparing groups and columns

Survival by sex: female 74.0%, male 18.9%.
Survival by class: class 1 62.6%, class 2 47.3%, class 3 24.2%.
Survival by sex and class: female, class 1 96.7%, female, class 2 92.1%, female, class 3 50.0%, male, class 1 36.9%, male, class 2 15.7%, male, class 3 13.5%.
These rates were calculated with Boolean masks, including `&` for joint groups.
An additional `|` mask for women **or** first-class passengers gives
63.6% survival; its groups overlap and therefore
should not be added together.

The heatmap uses exactly survived, pclass, age, sibsp, parch and fare. The two
largest absolute off-diagonal correlations are
pclass vs fare
(-0.548) and
sibsp vs parch
(+0.415). The sign shows whether each pair tends
to move together or in opposite directions. The negative value means lower numeric class codes (higher travel classes) tend to have higher fares.
The positive value means passengers traveling with more siblings/spouses also tend to have more parents/children on board. Correlation alone does not prove a cause of survival.

## What the four charts show

1. **Sex and class:** Survival is 96.7% for first-class women and
13.5% for third-class men. The gap shows that both recorded sex and
passenger class are useful signals. The chart alone does not prove why the gap happened.

2. **Age, class and survival:** Median age is 28.0
among survivors and 28.0 among non-survivors
in the EDA view. The overlapping boxes show that age alone does not cleanly
separate outcomes. Class changes the age mix within both groups.

3. **Fare, sex and survival:** Median fare is 26.00
for survivors and 10.50 for non-survivors.
The plot hides individual extreme points so the central boxes can be compared;
the separate fare box plot still counts the outliers. Fare likely reflects class
as well as other booking details.

4. **Family size and class:** The plot compares solo passengers, groups of 2-4,
and groups of 5+ within each class. The actual survival rates are
alone, class 1 52.3%, alone, class 2 34.6%, alone, class 3 21.3%, 2-4, class 1 73.3%, 2-4, class 2 62.8%, 2-4, class 3 40.7%, 5+, class 1 66.7%, 5+, class 2 100.0%, 5+, class 3 7.4%.
Both family size and class vary across groups, so these rates describe the sample and do not prove cause and effect.
Some family groups are small, so their percentages can be unstable.

## EDA scaling check

- age: mean 29.3152 -> 0.000000; population standard deviation 12.9776 -> 1.000000.
- fare: mean 32.0967 -> 0.000000; population standard deviation 49.6695 -> 1.000000.

These full-EDA z-scores are only a display check; modeling fits its own scaler
on the training split.
