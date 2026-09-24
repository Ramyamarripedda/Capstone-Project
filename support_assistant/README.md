# Policy support assistant

This is a small FastAPI app for the eight policy documents given in Q.docx.
It uses the assignment text, not checked current company policy.
See [local run steps](../LOCAL_RUN.md) for copy-and-paste commands.

## How a question moves through the app

1. `ingest.py` reads the eight files in `docs/`. Each file is one short chunk.
2. MiniLM turns each chunk into a list of numbers called an embedding.
   Chroma stores these numbers in `zepto_policy_chunks` and uses cosine distance.
3. `classify_intent` in `graph.py` checks the question. The required keywords
   send it to `retrieve_and_answer`; other questions go to `direct_answer`.
4. A policy question is embedded and compared with the stored chunks.
   The three closest chunks are retrieved. The mock answer uses the first
   200 characters of the closest chunk.
5. `schemas.py` checks the answer, source IDs and confidence. `main.py` returns
   them from POST `/ask` as JSON.

The LangGraph state is a TypedDict. The graph has the three named nodes and
a conditional edge after classification. The keyword rule is deliberately
simple: a question about an item may be missed if it has none of the listed
keywords. It follows the assignment's exact default rule.

## Mock and optional real-LLM modes

Unset `MOCK_LLM`, or set it to `1`, for the required mode. It makes no LLM
provider call. Embedding and retrieval still run for real on your computer.
After the first download, the model is saved in `model_cache/` and loaded
locally. Confidence is fixed at 1.0 by the assignment and is not a measured
answer-accuracy score.

Only `MOCK_LLM=0` enables the optional provider route. It needs `GROQ_API_KEY`;
`GROQ_MODEL` can choose a supported model. The classification and answer nodes
then call the provider. Retrieval remains local. `prompts.py` contains the
role, context, task, format and length instructions, a negative constraint and
a few-shot example. Invalid answer JSON gets at most two more attempts.
After three invalid answers the app returns a clear error answer.

## Recorded local examples

Policy request:

```json
{"query": "What is the delivery fee?"}
```

Response:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": [
    "doc_01",
    "doc_05",
    "doc_02"
  ],
  "confidence": 1.0
}
```

General request:

```json
{"query": "What is the capital of France?"}
```

Response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

The first result is `doc_01`, the delivery policy. The policy answer may stop
mid-sentence because the required mock template takes the first 200 characters.
`demo.py` checks the graph and endpoint; its results are in `outputs/`.

The Dockerfile installs CPU PyTorch, downloads the model and builds the index.
The local server and Docker container are tested with both request types.
No cloud deployment is required.
