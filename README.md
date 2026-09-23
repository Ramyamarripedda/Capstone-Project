# Capstone Project_AIML

Student-level starter structure for the Zepto Data & AI Platform described in Q.docx.
All three modules belong to this one repository.

**Status: project scaffold, not a completed submission.** The notebooks run their
setup and starter examples without installing packages or contacting external services.
They contain numbered work sections for the remaining assignment. A successful
starter notebook run does not mean the assignment criteria are complete.

## Folder structure

```text
Capstone_Project_AIML/
|-- README.md
|-- requirements.txt                 # notebook tools
|-- SUBMISSION_CHECKLIST.md
|-- ENVIRONMENT.md
|-- data_pipeline/
|   |-- 01_data_pipeline.ipynb
|   |-- README.md
|   |-- requirements.txt
|   |-- data/                       # scraped/cleaned data and SQLite database
|   `-- outputs/                    # actual SQL and pandas results
|-- analytics/
|   |-- 01_eda.ipynb
|   |-- 02_modeling.ipynb
|   |-- README.md
|   |-- requirements.txt
|   |-- data/                       # derived data, never a second raw download
|   |-- models/                     # complete fitted joblib pipeline
|   `-- outputs/                    # actual metrics and interpretations
|   # titanic.csv belongs directly here once downloaded
`-- support_assistant/
    |-- 01_support_assistant.ipynb
    |-- README.md
    |-- requirements.txt
    |-- docs/                       # exact eight supplied policy documents
    `-- outputs/                    # actual API example JSON responses
```

## Run the starter notebooks in VS Code

1. Open this project folder in VS Code.
2. Open a notebook and choose **Select Kernel > Python Environments > AI**.
3. Run all cells from top to bottom. The notebook checks the interpreter and package
   versions and runs the starter code. Missing future dependencies are reported.
4. Work through the numbered TODO sections and write your own interpretations.

Run order: `data_pipeline/01_data_pipeline.ipynb`, `analytics/01_eda.ipynb`,
`analytics/02_modeling.ipynb`, then `support_assistant/01_support_assistant.ipynb`.
The starter notebooks have been checked in Python 3.12.9. See ENVIRONMENT.md for
the observed packages and the limitations of that environment.

## Install full module dependencies later

Use Python 3.12. Existing AI packages have conflicts, so a separate project
environment is recommended before installing the full assignment stack:

```powershell
conda create -n capstone_aiml python=3.12 pip -y
conda activate capstone_aiml
python -m pip install -r requirements.txt
python -m pip install -r data_pipeline/requirements.txt
python -m pip install -r analytics/requirements.txt
python -m pip install -r support_assistant/requirements.txt
python -m pip check
```

Select `capstone_aiml` as the notebook kernel after installation. Dependencies use
major-version bounds; this is not a guarantee of compatibility or a security audit.
The complete dependency set has not yet been installed and tested together.
After end-to-end implementation and a clean `pip check`, save an exact lock snapshot
from this dedicated environment with `python -m pip freeze > requirements-lock.txt`.

## Design decisions

- Data pipeline: requests + BeautifulSoup, pandas and SQLite. Use the exact
  assignment constant **1 GBP = 105.50 INR**, not a live exchange rate.
- Analytics: one raw Titanic load and one committed `analytics/titanic.csv` fallback.
  Keep an EDA copy separate from training-only preprocessing to avoid leakage.
- Support assistant: eight supplied documents; default `MOCK_LLM=1`. Real local
  embeddings and ChromaDB retrieval still need implementation. The MiniLM model
  must be downloaded once and cached before offline embedding use.
- Keep code simple and put explanations in notebook Markdown or README files.

No secrets, local environments, original Word document, or generated model caches
are included. Required output files can be committed after they are produced and
reviewed. The example policy corpus is assignment text, not verified live policy.

## Submission and Git workflow

Use SUBMISSION_CHECKLIST.md to finish the required work. Submit only this public
repository link after completing the checklist. Review and understand the code,
and author your own reasoning and interpretations as required by Q.docx.

This scaffold is developed on `feature/project-structure` with two meaningful
commits and a merge into `main`. Preserve the merge history; do not squash it when
using it as evidence for the assignment's feature-branch workflow.
