# Run the project locally

## This computer: start the support app

The project environment already exists at `.venv`. Use it directly; activating
the older AI environment alone does not install the missing project packages.

```powershell
cd D:\Ramya_AIML\Capstone_Project_AIML
$env:MOCK_LLM = "1"
.\.venv\Scripts\python.exe -m uvicorn main:app --app-dir support_assistant --host 127.0.0.1 --port 7860
```

Open http://127.0.0.1:7860/docs. Select POST /ask, Try it out, then Execute.
The root address http://127.0.0.1:7860 also opens this test screen.

Policy question:

```json
{"query": "What is the delivery fee?"}
```

General question:

```json
{"query": "Hello"}
```

The response contains `answer`, `sources` and `confidence`. The policy answer
begins with `Based on the retrieved context:`. The general answer has no sources.
In mock mode confidence is always 1.0 by the assignment rule; it is not a
measured probability that the answer is correct.

Keep the server terminal open. Use Ctrl+C to stop it. To send a request from
a second PowerShell terminal:

```powershell
$body = @{query = "What is the delivery fee?"} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:7860/ask" -Method Post -ContentType "application/json" -Body $body
```

## Run notebooks in VS Code

Open a notebook. Click Select Kernel, then Python Environments, and select:

```text
D:\Ramya_AIML\Capstone_Project_AIML\.venv\Scripts\python.exe
```

If it is not listed, choose Select Another Kernel or Enter interpreter path.
Run `data_pipeline/01_data_pipeline.ipynb`, `analytics/01_eda.ipynb`,
`analytics/02_modeling.ipynb`, and `support_assistant/01_support_assistant.ipynb`.
Use Run All inside each notebook. The model-training notebook can take a few
minutes because it tries several Random Forest settings.

## First setup on another computer

Install Python 3.12 and Git, clone the repository, and create a fresh environment.
These commands are for a new checkout, not for replacing an existing environment.

```powershell
git clone https://github.com/Ramyamarripedda/Capstone-Project.git
cd Capstone-Project
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r data_pipeline/requirements.txt -r analytics/requirements.txt -r support_assistant/requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe support_assistant/ingest.py
```

The first install and model download need internet. `ingest.py` saves the model
under `support_assistant/model_cache/` and vectors under `chroma_db/`.
These large generated folders stay out of Git. The saved classifier was created
with scikit-learn 1.9.0, which is pinned in the analytics requirements. Re-run
modeling if you deliberately change that version instead of reusing its old file.

## Run with Docker

Start Docker Desktop. From the repository root:

```powershell
docker build -t zepto-support ./support_assistant
docker run --rm -p 127.0.0.1:7860:7860 zepto-support
```

Open http://127.0.0.1:7860/docs. The first build needs internet to install
packages and download MiniLM. The image stores the model and index so its
default mock requests can run offline. Stop it with Ctrl+C.

## Common problems

| Problem | What to do |
| --- | --- |
| ModuleNotFoundError | Use the project `.venv` Python, not the base/AI interpreter. |
| Port 7860 is in use | Stop the old server, or use `--port 7861` and open that port. |
| Notebook kernel stops | Select the project kernel and restart it; the support notebook runs heavy code in separate processes. |
| Model download fails | Check internet, run `support_assistant/ingest.py` again, then restart the app. |
| GET /ask gives 405 | `/ask` accepts POST. Use `/docs` to send the JSON body. |
| Empty query gives 422 | Enter a non-empty question. |
| Docker cannot connect | Start Docker Desktop and wait until its engine is running. |

Optional real-LLM mode is separate: it needs `MOCK_LLM=0` and a provider key.
The assignment can be completed using the default mock mode.
