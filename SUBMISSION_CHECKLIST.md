# Assignment checklist

Checked items were implemented and locally verified unless a limitation is
called out below. Review the generated interpretations in your own words.

## Repository

- [x] One public repository with `/data_pipeline`, `/analytics`,
  `/support_assistant`, root README and module requirements.
- [x] Four executable module notebooks plus readable `.py` implementations.
- [x] Text write-ups in Markdown and actual numeric outputs, not screenshots.
- [x] Feature branch with two commits and visible merge into main.
- [ ] Final personal review of all analysis text and code before submission.

## Data pipeline

- [x] 77 books scraped from three categories with requests/BeautifulSoup.
- [x] Title, listed GBP price, text rating, availability, category captured.
- [x] GBP float, rating integer, stock Boolean, INR at fixed 105.50.
- [x] Parse-failure choice explained; actual dropped count reported.
- [x] Regenerable two-table SQLite database with primary/foreign keys.
- [x] Six executed SQL queries cover all required clauses including JOIN.
- [x] At least two `pd.read_sql` results and equal `pd.merge` JOIN output.

## Analytics

- [x] Single Seaborn raw load and committed `analytics/titanic.csv` fallback.
- [x] Raw `info`, `describe`, shape, class balance and all missing rates.
- [x] Threshold-based cleaning decisions and train-only modeling imputation.
- [x] Age/fare histograms, boxplots, IQR counts and fare mean/median/mode.
- [x] Boolean-mask survival rates by sex, pclass and both together.
- [x] Exact six-variable correlation heatmap and top two absolute pairs.
- [x] Four multivariate charts with written interpretations.
- [x] Exploratory age/fare z-score before/after check.
- [x] Stratified split before train-only imputation, encoding and scaling.
- [x] Three classifiers on the same split, labeled tree, confusion matrices,
  accuracy, precision, recall, F1, ROC curves and AUC.
- [x] Baseline, balanced weights and train-only SMOTE comparison.
- [x] GridSearchCV over required forest parameters and OOB score.
- [x] Fare regression with MAE, RMSE, R2, adjusted R2, residual plot and
  heteroscedasticity conclusion.
- [x] Separate metric groups, classifier recommendation and reloaded complete
  fitted joblib pipeline predicting on raw inputs.

## Support assistant

- [x] Exact eight supplied documents indexed with local MiniLM/Chroma cosine.
- [x] Structured five-part prompt, negative constraint and few-shot example.
- [x] TypedDict LangGraph with required three nodes and conditional routing.
- [x] Exact default keyword rule, real top-three retrieval, templated mock
  policy answer and fixed general answer without provider calls.
- [x] Pydantic answer/sources/confidence and optional real-output retries.
- [x] FastAPI POST `/ask`, locally served, with two actual JSON examples.
- [x] Dockerfile and build/run instructions present.
- [x] Docker image built and container POST endpoint tested locally.
- [x] Architecture description covers ingestion, embedding, retrieval,
  generation, components and MOCK_LLM branches.
