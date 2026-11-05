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
