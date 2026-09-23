# Zepto Data and AI Platform

One repository for the three modules in Q.docx: a books data pipeline, Titanic
analytics, and a document-grounded Zepto policy assistant. The policy text is
the assignment corpus, not verified current Zepto policy.

## Setup and run

Use Python 3.12. The root `requirements.txt` covers notebook tools; each
module has its own `requirements.txt`. A fresh project environment is
recommended because the existing AI environment has unrelated package
conflicts (see `ENVIRONMENT.md`).

```powershell
conda create -n capstone_aiml python=3.12 pip -y
conda activate capstone_aiml
python -m pip install -r requirements.txt
python -m pip install -r data_pipeline/requirements.txt
python -m pip install -r analytics/requirements.txt
python -m pip install -r support_assistant/requirements.txt
python -m pip check
```

Select this Python environment as the VS Code notebook kernel and **Run All**
in this order:

1. `data_pipeline/01_data_pipeline.ipynb`
2. `analytics/01_eda.ipynb`
3. `analytics/02_modeling.ipynb`
4. `support_assistant/01_support_assistant.ipynb`

The notebooks run the readable `.py` scripts and display actual outputs.
Equivalent script commands from the repository root:

```powershell
python data_pipeline/pipeline.py
python analytics/eda.py
python analytics/modeling.py
python support_assistant/ingest.py
cd support_assistant
python -m uvicorn main:app --host 127.0.0.1 --port 7860
```

The books script needs the public scraping-practice site. EDA calls
`sns.load_dataset('titanic')` only if the committed `analytics/titanic.csv`
is absent. The support assistant downloads the public MiniLM model once and
then runs its default mock LLM mode without a provider key.

## Design decisions

- **Data pipeline:** scrape all pages of Mystery, Historical Fiction and
  Classics with requests and BeautifulSoup; drop invalid required fields;
  convert prices at the assignment constant **1 GBP = 105.50 INR**. SQLite
  uses `categories` and `books` tables connected by a foreign key. Six SQL
  query outputs and the matching pandas merge are in
  `data_pipeline/outputs/query_results.md`.
- **Analytics:** save the raw Titanic data once. The EDA copy fills age for
  descriptive plots; the shared clean CSV keeps missing age values so each
  modeling pipeline learns imputation on training data only. Charts and
  interpretations are in `analytics/outputs/`. The fitted complete pipeline
  is in `analytics/models/`.
- **Support assistant:** embed eight short supplied documents with local
  `all-MiniLM-L6-v2`; index them in Chroma with cosine distance; route policy
  and general questions through LangGraph; return Pydantic-validated JSON
  through FastAPI. `MOCK_LLM=1` is the default. The optional real-LLM path
  needs `GROQ_API_KEY`. Docker serves the same app locally.

The analysis text is a worked interpretation of measured outputs. Review and
rewrite it in your own words before submission, as Q.docx requires your own
reasoning. `SUBMISSION_CHECKLIST.md` maps all assignment requirements to the
files. Submit **this one public GitHub repository link** after review. The
earlier `feature/project-structure` branch has two commits and a visible
merge to `main`, satisfying the project-wide Git history requirement.
