"""LangGraph intent router with real retrieval and default deterministic answers."""

from __future__ import annotations

import json
import os
from typing import TypedDict

import httpx
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from ingest import retrieve
from prompts import GENERAL_PROMPT, POLICY_PROMPT
from schemas import AskResponse

KEYWORDS = ("delivery", "return", "refund", "membership", "tracking",
            "cancel", "gift card", "support hours")
GENERAL_ANSWER = "I can only answer questions about Zepto policies right now."


class SupportState(TypedDict, total=False):
    query: str
    intent: str
    retrieved: list[dict]
    response: dict


def mock_mode() -> bool:
    return os.getenv("MOCK_LLM", "1") != "0"


def real_llm(prompt: str) -> str:
    """Optional Groq free-tier route. Only called with MOCK_LLM=0."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Set GROQ_API_KEY to use optional MOCK_LLM=0 mode")
    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
              "messages": [{"role": "user", "content": prompt}],
              "temperature": 0},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def validated_real_answer(prompt: str, allowed_sources: set[str]) -> AskResponse:
    """Validate optional LLM output; make at most two corrective retries."""
    current = prompt
    for attempt in range(3):
        try:
            raw = real_llm(current)
            parsed = AskResponse.model_validate(json.loads(raw))
            if not set(parsed.sources).issubset(allowed_sources):
                raise ValueError("sources must be retrieved document IDs")
            return parsed
        except (ValueError, ValidationError, KeyError) as error:
            if attempt == 2:
                break
            current = (prompt + "\nCORRECTION: Previous response was invalid: "
                       + str(error) + ". Return only valid JSON with allowed sources.")
    return AskResponse(answer="ERROR: real LLM response failed schema validation after three attempts.",
                       sources=[], confidence=0.0)


def classify_intent(state: SupportState) -> SupportState:
    query = state["query"]
    if mock_mode():
        intent = ("policy_question" if any(word in query.lower() for word in KEYWORDS)
                  else "general_question")
    else:
        raw = real_llm("Classify this query. Reply with exactly policy_question or "
                       "general_question.\nQuery: " + query).strip().lower()
        intent = "policy_question" if raw == "policy_question" else "general_question"
    return {"intent": intent}


def retrieve_and_answer(state: SupportState) -> SupportState:
    found = retrieve(state["query"], n=3)  # Always real local embedding + Chroma search.
    if not found:
        raise RuntimeError("No policy chunks retrieved")
    if mock_mode():
        result = AskResponse(
            answer="Based on the retrieved context: " + found[0]["text"][:200],
            sources=[item["id"] for item in found], confidence=1.0)
    else:
        context = "\n".join(f"[{item['id']}] {item['text']}" for item in found)
        prompt = POLICY_PROMPT.format(context=context, question=state["query"])
        result = validated_real_answer(prompt, {item["id"] for item in found})
    return {"retrieved": found, "response": result.model_dump()}


def direct_answer(state: SupportState) -> SupportState:
    if mock_mode():
        result = AskResponse(answer=GENERAL_ANSWER, sources=[], confidence=1.0)
    else:
        prompt = GENERAL_PROMPT.format(question=state["query"])
        result = validated_real_answer(prompt, set())
    return {"response": result.model_dump()}


def route(state: SupportState) -> str:
    return state["intent"]


builder = StateGraph(SupportState)
builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("direct_answer", direct_answer)
builder.add_edge(START, "classify_intent")
builder.add_conditional_edges("classify_intent", route,
                              {"policy_question": "retrieve_and_answer",
                               "general_question": "direct_answer"})
builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)
graph = builder.compile()
