"""Request and validated response shapes for the support service."""

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    query: str = Field(min_length=1)


class AskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0, le=1)
