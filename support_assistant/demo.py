"""Run two default-mode graph/API examples and verify retrieval with real vectors."""

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from graph import graph
from ingest import collection, retrieve
from main import app
from schemas import AskResponse

ROOT = Path(__file__).resolve().parent
os.environ["MOCK_LLM"] = "1"


def run() -> dict:
    indexed = collection().count()
    if indexed != 8:
        raise AssertionError(f"Expected 8 indexed chunks; got {indexed}")
    matches = retrieve("What is the delivery fee?")
    if matches[0]["id"] != "doc_01":
        raise AssertionError(f"Delivery query matched {matches[0]['id']} first")
    client = TestClient(app)
    examples = {}
    for name, question, expected_intent in (
        ("policy", "What is the delivery fee?", "policy_question"),
        ("general", "What is the capital of France?", "general_question"),
    ):
        state = graph.invoke({"query": question})
        response = client.post("/ask", json={"query": question})
        response.raise_for_status()
        parsed = AskResponse.model_validate(response.json())
        if state["intent"] != expected_intent:
            raise AssertionError(f"Wrong route for {question}")
        examples[name] = {"intent": state["intent"], "response": parsed.model_dump()}
    summary = {"indexed_chunks": indexed,
               "delivery_top_three": [item["id"] for item in matches],
               "examples": examples}
    (ROOT / "outputs" / "demo_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
