from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from .base import ConfidenceLevel, PatternType

class Pattern(BaseModel):
    type: PatternType = Field(..., description="Type of pattern detected")
    description: str = Field(..., description="Description of the pattern")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for this pattern")
    evidence: List[str] = Field(..., description="Evidence supporting this pattern")

class AnalysisReport(BaseModel):
    patterns: List[Pattern] = Field(..., description="All detected patterns")
    structural_features: Dict[str, Any] = Field(
        ...,
        description="Structural features like lists, headers, formatting"
    )
    linguistic_features: Dict[str, Any] = Field(
        ...,
        description="Linguistic features like tone, formality, vocabulary"
    )
    constraints_detected: List[str] = Field(
        ...,
        description="Detected constraints like length, format, inclusions"
    )
    overall_confidence: ConfidenceLevel = Field(
        ...,
        description="Overall confidence in the analysis"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "patterns": [
                    {
                        "type": "structural",
                        "description": "Numbered list format",
                        "confidence": 0.95,
                        "evidence": ["Output contains numbered items 1-5"]
                    }
                ],
                "structural_features": {
                    "has_list": True,
                    "list_type": "numbered",
                    "item_count": 5
                },
                "linguistic_features": {
                    "tone": "informative",
                    "formality": "neutral",
                    "domain": "health"
                },
                "constraints_detected": [
                    "Exactly 5 items",
                    "Brief descriptions",
                    "Health benefits focus"
                ],
                "overall_confidence": "high"
            }
        }