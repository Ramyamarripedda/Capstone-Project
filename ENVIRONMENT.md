# Environment and verification

Python 3.12.9 from the existing AI environment was used to create a
project-local `.venv` with `--system-site-packages`. Missing packages were
installed in `.venv`, leaving AI itself unchanged. This local overlay can see
AI's pre-existing package conflicts, so a clean project environment is the
recommended reproducible setup described in the root README.

The run used imbalanced-learn 0.14.2, sentence-transformers 5.7.0,
datasets 4.8.5, ChromaDB 1.5.8 and LangGraph 1.2.12 in `.venv`; it used
AI's Python 3.12.9, pandas 3.0.3, scikit-learn 1.9.0, seaborn 0.13.2 and
PyTorch 2.6.0. The module requirements give compatible version ranges, not
an exact lockfile. `pip check` in AI was not clean even before this work;
existing unrelated packages require conflicting pandas, protobuf and other
versions. A new environment should be installed and checked as a unit.

Verified on this machine:

- Live practice-site scrape: 77 clean books in three categories, SQLite
  foreign keys valid, six SQL queries executed, pandas merge equal to SQL JOIN.
- Titanic raw dataset downloaded once through Seaborn and committed as CSV;
  EDA charts/results generated from 891 raw rows. Modeling ran on 889 rows,
  including SMOTE, Random Forest tuning, fare regression and joblib reload.
- MiniLM downloaded and embedded all eight documents in ChromaDB. Local
  uvicorn served two HTTP POST `/ask` requests with default mock mode and
  returned validated JSON. The policy example retrieved `doc_01` first.
- Docker image built with CPU PyTorch, and a container served both POST
  `/ask` examples on local host port 7861 with matching JSON responses.
- All four notebooks passed `nbformat` validation and full top-to-bottom
  execution with the project Python kernel. The support notebook launches
  embedding and API checks in separate processes to keep Jupyter stable on
  this Windows machine.
- The optional real LLM path was implemented but not called; it requires an
  API key and is not part of the graded baseline.

Docker Desktop's Linux engine was started for the container check. The
temporary test container was stopped after validation.

Official installation references:
https://scikit-learn.org/stable/install and
https://sbert.net/docs/installation.html.
