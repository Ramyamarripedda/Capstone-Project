# AIML Capstone Project

This project has three parts. They show how to collect data, study data and
build a small question-answer service. All three parts are in this repository.

| Folder | What it does | Main notebook |
| --- | --- | --- |
| data_pipeline | Scrapes book details, cleans them and saves them in SQLite | 01_data_pipeline.ipynb |
| analytics | Studies Titanic data and trains prediction models | 01_eda.ipynb, then 02_modeling.ipynb |
| support_assistant | Finds information in eight policy files and returns an answer | 01_support_assistant.ipynb |

The book data, Titanic data and policy text are three separate practice datasets.
They are not combined into one training dataset.

## Run the app on this computer

Open a PowerShell terminal and run:

```powershell
cd D:\Ramya_AIML\Capstone_Project_AIML
.\.venv\Scripts\python.exe -m uvicorn main:app --app-dir support_assistant --host 127.0.0.1 --port 7860
```

Open **http://127.0.0.1:7860/docs** in your browser. Click **POST /ask**, then
**Try it out**. Enter the following JSON and click **Execute**:

```json
{"query": "What is the delivery fee?"}
```

Try `{"query": "Hello"}` to see the general-question response.
Keep the terminal open while using the app. Press **Ctrl+C** to stop it.
The app is a FastAPI service; `/docs` is its built-in test screen.
See [LOCAL_RUN.md](LOCAL_RUN.md) for setup on another computer, notebooks,
Docker commands and common errors.

## Run all three parts

Use the project `.venv` Python as the VS Code notebook kernel. Run the four
notebooks in the table from top to bottom. The notebook code calls the small
Python files in the same folder. The support notebook uses separate Python
processes because direct model imports caused a kernel crash on this Windows
installation.

You can also run these scripts from the repository root:

```powershell
.\.venv\Scripts\python.exe data_pipeline/pipeline.py
.\.venv\Scripts\python.exe analytics/eda.py
.\.venv\Scripts\python.exe analytics/modeling.py
.\.venv\Scripts\python.exe support_assistant/demo.py
```

The books script needs internet. Titanic is saved in `analytics/titanic.csv`
for offline use. The MiniLM embedding model downloads once; later calls load
the saved local copy. Default `MOCK_LLM=1` does not call an LLM provider and
does not need an API key.

## Main choices

- The book pipeline uses a fixed rate of **1 GBP = 105.50 INR**. Invalid rows
  are dropped and counted. The last run had 77 valid books and zero dropped rows.
- Titanic cleaning follows the missing-value rules in the assignment. EDA
  uses an age-filled copy for plots. Model preprocessing is fitted only on
  training rows. The model is selected using training cross-validation.
- The assistant uses real local embeddings and Chroma retrieval. Its default
  answer is a fixed template containing a short policy excerpt. It is not a
  free-form AI chatbot in mock mode. The policy text comes from the assignment.

## Files to read

- [Data pipeline notes](data_pipeline/README.md) and `outputs/query_results.md`.
- [Analytics notes](analytics/README.md), `outputs/eda_interpretation.md` and
  `outputs/model_interpretation.md`.
- [Support assistant notes](support_assistant/README.md) and its saved JSON responses.
- [Requirement checklist](SUBMISSION_CHECKLIST.md) and [review notes](REVIEW.md).

The root requirements file installs notebook tools; each module has its own
requirements file. [ENVIRONMENT.md](ENVIRONMENT.md) records the tested setup.
Submit this one repository link after reading the code and checking that the
written explanations match your understanding. The separate Telugu PDF is a
learning guide and is not part of the submission repository.
