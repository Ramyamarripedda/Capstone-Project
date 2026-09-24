# Final project review

All three modules were checked against Q.docx. They use separate practice
datasets: books, Titanic passengers and eight supplied policy documents.

## Improvements

- README files and result explanations use simple English.
- The saved classifier is chosen using five-fold training cross-validation
  F1. Test scores are used for the final comparison.
- Confusion-matrix counts are saved in Markdown and JSON; ROC points are
  saved as CSV files.
- Fare regression drops the first encoded category when counting predictors
  for adjusted R2.
- The scraper handles missing fields and both stock states. SQLite
  connections close after use to avoid locked files on Windows.
- Both optional LLM prompts format correctly. Tests cover JSON retries
  without calling a provider.
- Empty questions are rejected. Opening the app root takes you to `/docs`.
- MiniLM is saved locally after its first download for offline mock use.

## Results and checks

- Books: 77 rows, three categories, six SQL queries and matching pandas JOIN.
- Titanic: 891 raw rows, 889 cleaned rows, 711 training and 178 test rows.
- Selected tuned Random Forest: training CV F1 about 0.767, test accuracy
  about 0.843 and test F1 about 0.788.
- Fare regression: MAE 21.100, RMSE 41.702, R2 0.348, adjusted R2 0.317.
- Support assistant: eight chunks; delivery retrieves `doc_01` first.
- All four notebooks passed full execution. All eight automated tests passed.
- Local FastAPI and the rebuilt Docker container served both example requests.

Run the tests from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

See [LOCAL_RUN.md](LOCAL_RUN.md) for app, notebook and Docker commands.
The optional provider LLM mode was not called. Mock confidence 1.0 is a fixed
assignment value, not a measured probability. The current local environment
inherits existing AI packages; clean Python 3.12 setup is documented for a
new computer. Requirements are not a complete lockfile.

The Telugu learning PDF is outside this repository because Q.docx asks for
code and Markdown submission. Read and understand the work before submitting;
the personal-review checkbox remains for the student.
