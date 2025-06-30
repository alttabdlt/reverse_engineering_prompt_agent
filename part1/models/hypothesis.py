from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PromptHypothesis(BaseModel):
    prompt: str = Field(..., description="The hypothesized prompt")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score for this hypothesis"
    )
    reasoning: str = Field(..., description="Explanation for why this prompt is likely")
    key_elements: List[str] = Field(
        ...,
        description="Key elements identified in the prompt"
    )
    rank: int = Field(..., ge=1, description="Ranking among all hypotheses")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "List 5 benefits of regular exercise",
                "confidence": 0.85,
                "reasoning": "The output is a numbered list of exactly 5 items about exercise benefits, suggesting a direct instruction to list benefits",
                "key_elements": ["List 5", "benefits", "exercise"],
                "rank": 1
            }
        }

class HypothesisSet(BaseModel):
    hypotheses: List[PromptHypothesis] = Field(
        ...,
        description="List of prompt hypotheses ordered by confidence"
    )
    analysis_context: Dict[str, Any] = Field(
        ...,
        description="Context from analysis that informed these hypotheses"
    )
    iteration: int = Field(..., description="Which iteration/pass generated these")