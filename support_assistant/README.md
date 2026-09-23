# Zepto policy support assistant

`docs/doc_01.txt` through `docs/doc_08.txt` contain the exact eight assignment
documents. These are example project policies. From this directory:

```powershell
python ingest.py
python -m uvicorn main:app --host 127.0.0.1 --port 7860
```

The first run downloads `all-MiniLM-L6-v2` once. Leave `MOCK_LLM` unset or
set it to `1` for the graded, keyless mode. `demo.py` runs the local graph/API
checks and saves `outputs/demo_results.json`. The notebook calls these scripts
and displays the prompt, retrieval results and graph routing.

Actual local uvicorn POST `/ask` examples with default mock mode:

```json
{"query":"What is the delivery fee?"}
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del","sources":["doc_01","doc_05","doc_02"],"confidence":1.0}
```

```json
{"query":"What is the capital of France?"}
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

The raw JSON response files are in `outputs/`. `doc_01` is the delivery
policy and was the top match for the policy example. The mock answer uses
the required first 200 characters of that chunk, so it demonstrates retrieval
rather than serving as a polished customer answer.

## Architecture

1. **Ingestion:** `ingest.py` reads each short document as one `doc_XX` chunk.
2. **Embedding:** the same file uses local MiniLM vectors and stores them in
   the persistent cosine-distance Chroma `zepto_policy_chunks` collection.
3. **Retrieval:** `graph.py` classifies intent with the exact keyword rule in
   default mode. LangGraph routes policy questions to `retrieve_and_answer`,
   which retrieves the top three chunks, and general questions to
   `direct_answer`, which does not retrieve.
4. **Generation:** default `MOCK_LLM=1` uses a top-chunk template or fixed
   string. `schemas.py` validates `answer`, `sources`, `confidence`. Optional
   `MOCK_LLM=0` calls a real LLM using the structured prompt in `prompts.py`;
   invalid JSON gets up to two corrective retries. Retrieval remains real and
   local in both modes.

## Docker

```powershell
docker build -t zepto-support .
docker run --rm -p 7860:7860 zepto-support
```

The Dockerfile installs CPU PyTorch and the other dependencies, downloads
MiniLM and builds the local index during image build, then serves FastAPI.
It was built and run locally; both example POST requests returned the same
valid JSON from the container on host port 7861. The uvicorn endpoint was
also tested outside Docker. Cloud deployment is optional.
