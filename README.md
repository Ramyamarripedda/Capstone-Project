# AIML Capstone Project

This project has three modules: collecting book data, analysing Titanic data,
and building a policy question-answer app. The notebooks, Python files and
results for each module are in the folders below.

## 1. Book data pipeline

The script collects books from three categories on Books to Scrape. It cleans
prices, ratings and stock details, then saves the data in CSV files and SQLite.
The price conversion uses the assignment rate: **1 GBP = 105.50 INR**.

The latest run collected 77 books. Six SQL queries are included, and the JOIN
result is compared with a pandas merge.

- [Notebook](data_pipeline/01_data_pipeline.ipynb)
- [SQL results](data_pipeline/outputs/query_results.md)
- [Module notes](data_pipeline/README.md)

## 2. Titanic analysis and models

This module checks missing values, makes charts and compares survival rates.
It then trains Logistic Regression, Decision Tree and Random Forest models.
There is also a separate model for predicting fare.

The cleaned dataset has 889 rows. The split uses 711 rows for training and
178 for testing. The tuned Random Forest was selected using training
cross-validation. Its test accuracy was about 84.3% and F1 was about 0.788.

- [EDA notebook](analytics/01_eda.ipynb)
- [Modeling notebook](analytics/02_modeling.ipynb)
- [EDA results](analytics/outputs/eda_interpretation.md)
- [Model results](analytics/outputs/model_interpretation.md)
- [Module notes](analytics/README.md)

## 3. Policy support assistant

This app uses the eight policy documents provided in the assignment.
MiniLM and Chroma find the relevant documents. LangGraph routes the question,
and FastAPI returns the answer with source IDs and confidence.

The default is mock mode. It returns a short excerpt from the closest
matching document, so an answer can stop in the middle of a sentence.
It does not need an LLM API key. The embedding model needs to download once.

- [Notebook](support_assistant/01_support_assistant.ipynb)
- [App explanation and Docker steps](support_assistant/README.md)
- [Example responses](support_assistant/outputs/demo_results.json)

## How to run

Use Python 3.12. From the repository folder, create an environment and install
the requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r data_pipeline/requirements.txt -r analytics/requirements.txt -r support_assistant/requirements.txt
```

If the project environment is already set up, skip those steps.
In VS Code, select `.venv` as the notebook kernel and use **Run All**.
Run the EDA notebook before the modeling notebook. The book scraper needs
internet; the Titanic CSV is included in the repository.

To start the support app from the repository folder:

```powershell
$env:MOCK_LLM = "1"
.\.venv\Scripts\python.exe -m uvicorn main:app --app-dir support_assistant --host 127.0.0.1 --port 7860
```

Open http://127.0.0.1:7860/docs. Click **POST /ask**, then **Try it out**.
Enter this request and click **Execute**:

```json
{"query": "What is the delivery fee?"}
```

A successful request shows code 200. Try `{"query": "Hello"}` for the other
response type. Press **Ctrl+C** in the terminal to stop the app.

More setup steps are in [LOCAL_RUN.md](LOCAL_RUN.md). The package versions
used for testing are listed in [ENVIRONMENT.md](ENVIRONMENT.md).

## Checks

All four notebooks ran successfully. The app was also tested locally and
inside Docker. The eight automated checks can be run with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The optional real LLM mode was not tested with a provider. The book, Titanic
and policy datasets are used for their own modules. They are not joined together.
