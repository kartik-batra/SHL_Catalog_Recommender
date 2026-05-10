from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1, description="Full conversation history")


class Recommendation(BaseModel):
    name: str = Field(..., description="Exact assessment name from catalog")
    url: str = Field(..., description="Exact catalog URL — never fabricated")
    test_type: str = Field(..., description="Space-separated type codes e.g. 'A K'")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent's conversational reply")
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="1–10 items when recommending; empty list otherwise",
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the agent considers the task complete",
    )


class AssessmentItem(BaseModel):
    """One row in catalog.json."""
    name: str
    url: str
    test_types: list[str] = Field(default_factory=list)
    remote_testing: bool = False
    adaptive_irt: bool = False
    description: str = ""
    job_levels: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
