"""Run after installing dependencies: python -m unittest discover -s tests -v."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))
sys.path.insert(0, str(ROOT / "support_assistant"))
from pipeline import clean_books, load_database, query_and_compare
import graph as support_graph
from main import app
from prompts import GENERAL_PROMPT, POLICY_PROMPT


class ProjectTests(unittest.TestCase):
    def test_bad_book_values_are_dropped(self):
        good = {"title": "Example", "price": "\u00a310.00", "star_rating": "Three",
                "availability": "In stock", "category": "Test category"}
        rows = [good, {**good, "price": "bad"}, {**good, "star_rating": "unknown"},
                {**good, "availability": "unknown"}, {**good, "title": ""}]
        clean, dropped = clean_books(pd.DataFrame(rows))
        self.assertEqual(dropped, 4)
        self.assertEqual(clean.iloc[0]["price_inr"], 1055.0)
        self.assertEqual(str(clean.rating.dtype), "int64")
        self.assertEqual(str(clean.in_stock.dtype), "bool")

    def test_saved_real_books_rebuild_and_join(self):
        raw = pd.read_csv(ROOT / "data_pipeline/data/books_raw.csv")
        clean, _ = clean_books(raw)
        self.assertGreaterEqual(len(clean), 60)
        self.assertGreaterEqual(clean.category.nunique(), 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "books.db"
            load_database(clean, path)
            results, matches = query_and_compare(path)
        self.assertTrue(matches)
        self.assertEqual(len(results["05_side_by_side"]), 10)

    def test_prompt_examples_format_correctly(self):
        self.assertIn("Hello", GENERAL_PROMPT.format(question="Hello"))
        self.assertIn("policy text", POLICY_PROMPT.format(context="policy text", question="delivery?"))

    def test_real_answer_retries_then_validates(self):
        valid = json.dumps({"answer": "Example", "sources": ["doc_01"], "confidence": 0.8})
        with patch.object(support_graph, "real_llm", side_effect=["not JSON", valid]) as call:
            result = support_graph.validated_real_answer("test prompt", {"doc_01"})
        self.assertEqual(call.call_count, 2)
        self.assertEqual(result.answer, "Example")

    def test_real_answer_stops_after_three_invalid_outputs(self):
        with patch.object(support_graph, "real_llm", return_value="not JSON") as call:
            result = support_graph.validated_real_answer("test prompt", set())
        self.assertEqual(call.call_count, 3)
        self.assertTrue(result.answer.startswith("ERROR:"))
        self.assertEqual(result.confidence, 0.0)

    def test_optional_general_route_formats_and_validates(self):
        valid = json.dumps({"answer": "Policies only", "sources": [], "confidence": 1.0})
        with patch.dict("os.environ", {"MOCK_LLM": "0"}), patch.object(
                support_graph, "real_llm", return_value=valid):
            state = support_graph.direct_answer({"query": "Hello"})
        self.assertEqual(state["response"]["sources"], [])

    def test_default_api_routes_and_never_calls_provider(self):
        client = TestClient(app)
        with patch.dict("os.environ", {"MOCK_LLM": "1"}), patch.object(
                support_graph, "real_llm", side_effect=AssertionError("Provider call in mock mode")):
            policy = client.post("/ask", json={"query": "What is the delivery fee?"})
            general = client.post("/ask", json={"query": "Hello"})
        self.assertEqual(policy.status_code, 200)
        self.assertEqual(policy.json()["sources"][0], "doc_01")
        self.assertEqual(len(policy.json()["sources"]), 3)
        self.assertTrue(policy.json()["answer"].startswith("Based on the retrieved context: "))
        self.assertEqual(general.status_code, 200)
        self.assertEqual(general.json()["sources"], [])
        self.assertEqual(client.post("/ask", json={"query": "   "}).status_code, 422)
        self.assertEqual(client.get("/", follow_redirects=False).headers["location"], "/docs")

    def test_model_choice_and_text_outputs(self):
        result = json.loads((ROOT / "analytics/outputs/model_results.json").read_text())
        best = max(result["training_cv_f1"], key=result["training_cv_f1"].get)
        self.assertEqual(result["best_classifier"], best)
        for matrix in result["confusion_matrices"].values():
            self.assertEqual(sum(sum(row) for row in matrix), result["test_rows"])
        self.assertTrue(result["saved_pipeline_reload_matches"])


if __name__ == "__main__":
    unittest.main()
