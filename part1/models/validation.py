from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ValidationResult(BaseModel):
    hypothesis: str = Field(..., description="The prompt hypothesis being validated")
    match_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall match score"
    )
    semantic_similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Semantic similarity between original and regenerated output"
    )
    structural_match: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How well the structure matches"
    )
    constraint_satisfaction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How well constraints are satisfied"
    )
    missing_elements: List[str] = Field(
        ...,
        description="Elements present in output but missing from hypothesis"
    )
    extra_elements: List[str] = Field(
        ...,
        description="Elements in hypothesis not reflected in output"
    )
    confidence_calibration: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How well calibrated the confidence is"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "hypothesis": "List 5 benefits of regular exercise",
                "match_score": 0.88,
                "semantic_similarity": 0.92,
                "structural_match": 0.95,
                "constraint_satisfaction": 0.90,
                "missing_elements": ["specific focus on health benefits"],
                "extra_elements": [],
                "confidence_calibration": 0.85
            }
        }