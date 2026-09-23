# Environment check

The starter notebooks were checked using the existing AI environment with Python
3.12.9. No packages were installed, upgraded, downgraded or removed from AI.

Observed packages: nbformat 5.10.4, nbclient 0.10.2, ipykernel 6.29.5,
requests 2.34.2, beautifulsoup4 4.13.3, numpy 2.2.6, pandas 3.0.3,
matplotlib 3.10.1, seaborn 0.13.2, scikit-learn 1.9.0, joblib 1.4.2,
fastapi 0.136.3, pydantic 2.13.4, uvicorn 0.34.0 and httpx 0.28.1.

Missing full-assignment dependencies: imbalanced-learn, sentence-transformers,
chromadb and langgraph. The starter notebooks report these without trying to
install them automatically.

`python -m pip check` reported pre-existing conflicts involving aiobotocore,
bqplot, covjsonkit, databricks-sdk, datasets, elapid, lightning, mlflow,
opentelemetry-proto and pynacl. For example, bqplot and mlflow require pandas below
3 while this environment has pandas 3.0.3. This is not a clean environment for
validating the entire capstone dependency set.

Starter execution is a limited check: it does not validate web scraping, Titanic
model training, embedding downloads, ChromaDB, LangGraph, the API or Docker.
All four notebooks passed nbformat validation and ran top to bottom through
nbclient using the registered AI kernel. EDA used its default offline starter
mode without a Titanic CSV. No notebook errors occurred. A Windows Jupyter
event-loop compatibility warning was emitted, but execution completed.
Use a fresh environment before adding the remaining packages. Installation
guidance: https://scikit-learn.org/stable/install and
https://sbert.net/docs/installation.html.
