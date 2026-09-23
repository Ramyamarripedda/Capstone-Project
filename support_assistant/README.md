# Support assistant

Open `01_support_assistant.ipynb`. It loads the exact eight policy texts and
demonstrates the required deterministic intent classifier. The notebook does
not yet implement embeddings, retrieval, LangGraph, FastAPI or Docker.

The final implementation should add `main.py` for FastAPI, `graph.py` for
LangGraph, `ingest.py` for local embedding/indexing, `prompts.py`, `schemas.py`,
and a working Dockerfile. These are future tasks, not empty executable files.
Store actual example responses in `outputs/`.

Default to MOCK_LLM=1. Initial package/model installation needs internet; cache
all-MiniLM-L6-v2 before testing offline. Mock LLM mode still requires genuine
local embedding and retrieval. Never commit API keys or local vector caches.

The documents are the exact assignment corpus, not current real-world policies.
