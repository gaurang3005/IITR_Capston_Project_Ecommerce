"""Pydantic contracts: every structured output in the project validates against one of these."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CrewResponse(BaseModel):
    """Every crew response must validate against this (Task 9)."""
    answer: str = Field(..., min_length=1, description="Customer-facing answer")
    sources: list[str] = Field(default_factory=list, description="kb-xx doc ids and ORDxxxx ids used")
    grounded: bool = Field(..., description="True if the answer is backed by KB text or an order record")
    escalated: bool = Field(..., description="True if a human must follow up")
    confidence: float = Field(..., ge=0.0, le=1.0)

class JudgeScore(BaseModel):
    """Task 13 — one LLM-as-judge verdict, each property 1-5."""
    accuracy: int = Field(..., ge=1, le=5)
    grounding: int = Field(..., ge=1, le=5)
    completeness: int = Field(..., ge=1, le=5)
    safety: int = Field(..., ge=1, le=5)
    rationale: str = ""

class ReviewVerdict(BaseModel):
    """Task 14 — the Autogen Final_Editor's structured output (field names fixed by the brief)."""
    approved: bool
    final_answer: str
    reason: str
