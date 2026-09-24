"""Run locally: python -m uvicorn main:app --host 127.0.0.1 --port 7860."""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from graph import graph
from schemas import AskRequest, AskResponse

app = FastAPI(title="Zepto policy assistant (assignment corpus)")


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = graph.invoke({"query": request.query})
    return AskResponse.model_validate(result["response"])
