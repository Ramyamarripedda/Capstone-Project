# Submission checklist

This checklist records future assignment work, not claimed results.

## Repository
- [x] One repository with the three required module folders and root README.
- [x] Four runnable starter notebooks and per-module dependency specifications.
- [ ] Complete and execute every module end to end; record genuine outputs.
- [ ] Finish written interpretations, run instructions and design decisions.
- [ ] Verify a feature branch has at least two commits and a visible merge to main.
- [ ] Review files for secrets and submit the single public repository link.

## Data pipeline
- [ ] Scrape at least 60 real books across at least 3 categories, handling pagination.
- [ ] Capture title, GBP price, text rating, availability and category.
- [ ] Produce price_gbp float, rating integer 1-5, in_stock boolean, price_inr.
- [ ] Justify parsing failures: median numeric imputation or dropping invalid rows.
- [ ] Apply exactly 105.50 INR per GBP and document it.
- [ ] Populate normalized categories/books tables with PK/FK and recreation code.
- [ ] Execute >=5 SQL queries covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT,
      IN or BETWEEN, and at least one JOIN. Save each query and actual output.
- [ ] Read >=2 queries using pd.read_sql; reproduce the JOIN with pd.merge and
      show matching results side by side.

## Analytics
- [ ] Load raw Titanic once with sns.load_dataset; immediately save titanic.csv.
- [ ] Report info, describe, shape, class balance and all missing percentages.
- [ ] Apply <5% row dropping, 5-30% imputation, and justify high-missing treatment.
- [ ] Keep EDA imputation separate from train-only modeling preprocessing.
- [ ] Age/fare histograms, boxplots and IQR outlier counts; fare mean/median/mode
      and a justified skewness conclusion.
- [ ] Boolean-mask survival rates by sex, pclass and sex+pclass.
- [ ] Exact 6-column correlation heatmap and top two unique absolute correlations.
- [ ] >=4 multivariate charts, each with 2-4 sentence interpretations.
- [ ] Age/fare EDA standardization before/after statistics, not fed into modeling.
- [ ] One stratified split before fitting modeling imputation/encoding/scaling.
- [ ] Logistic Regression, Decision Tree, Random Forest on the same split.
- [ ] Labeled plot_tree; confusion matrices, accuracy, precision, recall, F1, ROC/AUC.
- [ ] Baseline vs balanced class weights vs training-fold-only SMOTE comparison.
- [ ] Random Forest GridSearchCV over n_estimators, max_depth, max_features;
      construct with oob_score=True and report best parameters and OOB score.
- [ ] Fare multivariate regression: MAE, RMSE, R2, adjusted R2 and residual plot
      with a written heteroscedasticity conclusion.
- [ ] Comparison table with separate classification/regression metric groups and
      3-5 sentence recommendation citing actual metrics.
- [ ] Save/reload complete joblib pipeline and verify predictions on raw inputs.

## Support assistant
- [x] Copy exact eight assignment document bodies into docs/doc_01.txt...doc_08.txt.
- [ ] Chunk, embed with all-MiniLM-L6-v2 and index in persistent ChromaDB.
- [ ] Actual role/context/task/format/length prompt, negative constraint, few-shot.
- [ ] TypedDict LangGraph with classify_intent, retrieve_and_answer, direct_answer
      and conditional routing using the exact keyword heuristic in default mode.
- [ ] Genuine top-3 cosine retrieval and top-chunk canned mock answer.
- [ ] Default MOCK_LLM=1 with no provider calls; unrelated questions use fixed text.
- [ ] Pydantic answer/sources/confidence schema; optional real-output validation
      retry logic with up to two additional attempts.
- [ ] FastAPI POST /ask and actual policy/general example JSON responses.
- [ ] Dockerfile successfully built and run locally with a working /ask endpoint.
- [ ] Architecture narrative names ingestion, embedding, retrieval, generation
      components and explains what MOCK_LLM changes.
