from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from .base import BaseResponse, ConfidenceLevel
from .hypothesis import PromptHypothesis
from .validation import ValidationResult

class PromptComparison(BaseModel):
    """Side-by-side comparison of prompts"""
    hypothesis_prompt: str = Field(..., description="The hypothesized prompt")
    simulated_output: str = Field(..., description="Output generated from hypothesis")
    original_output: str = Field(..., description="The original output to match")
    differences: List[str] = Field(default_factory=list, description="Key differences identified")
    similarities: List[str] = Field(default_factory=list, description="Key similarities identified")

class EnhancedScoring(BaseModel):
    """Enhanced scoring breakdown"""
    semantic_similarity: float = Field(..., description="Semantic similarity score (0-1)")
    structural_match: float = Field(..., description="Structural match score (0-1)")
    constraint_satisfaction: float = Field(..., description="Constraint satisfaction score (0-1)")
    style_match: float = Field(..., description="Style and tone match score (0-1)")
    intent_preservation: float = Field(..., description="Intent preservation score (0-1)")
    complexity_penalty: float = Field(..., description="Penalty for overly complex prompts (0-1)")
    
    # Component breakdowns
    semantic_components: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of semantic similarity components"
    )
    structural_components: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of structural match components"
    )
    
    # Confidence metrics
    confidence_score: float = Field(..., description="Overall confidence score (0-1)")
    confidence_interval: Tuple[float, float] = Field(
        ...,
        description="95% confidence interval for the score"
    )
    uncertainty_factors: List[str] = Field(
        default_factory=list,
        description="Factors contributing to uncertainty"
    )
    
    # Adaptive weights used
    weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Dynamic weights used for this evaluation"
    )

class DetectionResult(BaseModel):
    best_hypothesis: PromptHypothesis = Field(
        ...,
        description="The most likely prompt hypothesis"
    )
    all_hypotheses: List[PromptHypothesis] = Field(
        ...,
        description="All generated hypotheses ranked by confidence"
    )
    validation_results: List[ValidationResult] = Field(
        ...,
        description="Validation results for each hypothesis"
    )
    confidence: ConfidenceLevel = Field(
        ...,
        description="Overall confidence in the detection"
    )
    
    # New comparison feature
    prompt_comparison: Optional[PromptComparison] = Field(
        None,
        description="Side-by-side comparison of best hypothesis"
    )
    
    # Enhanced scoring
    enhanced_scoring: Optional[EnhancedScoring] = Field(
        None,
        description="Detailed scoring breakdown with advanced metrics"
    )
    
    attempts_used: int = Field(..., description="Number of refinement attempts used")
    execution_trace: List[Dict[str, Any]] = Field(
        ...,
        description="Trace of the detection process"
    )
    thinking_process: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="The agent's thinking process at each stage"
    )
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")

class AnalyzeResponse(BaseResponse):
    result: Optional[DetectionResult] = Field(
        None,
        description="The detection result if successful"
    )
    request_id: str = Field(..., description="Unique request identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Successfully detected prompt",
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_123456",
                "result": {
                    "best_hypothesis": {
                        "prompt": "List 5 benefits of regular exercise",
                        "confidence": 0.88,
                        "reasoning": "Output structure and content strongly suggest a direct list request",
                        "key_elements": ["List 5", "benefits", "exercise"],
                        "rank": 1
                    },
                    "all_hypotheses": [],
                    "validation_results": [],
                    "confidence": "high",
                    "prompt_comparison": {
                        "hypothesis_prompt": "List 5 benefits of regular exercise",
                        "simulated_output": "1. Improves heart health\n2. Boosts mood\n3. Helps weight management\n4. Increases energy\n5. Better sleep",
                        "original_output": "1. Improves cardiovascular health\n2. Boosts mood and mental health\n3. Helps with weight management\n4. Increases energy levels\n5. Promotes better sleep",
                        "differences": ["Original has more detailed descriptions"],
                        "similarities": ["Both have 5 numbered items", "Similar topics covered"]
                    },
                    "enhanced_scoring": {
                        "semantic_similarity": 0.92,
                        "structural_match": 0.95,
                        "constraint_satisfaction": 1.0,
                        "style_match": 0.88,
                        "intent_preservation": 0.94,
                        "complexity_penalty": 0.85,
                        "semantic_components": {
                            "embedding": 0.91,
                            "ngram": 0.89,
                            "concepts": 0.95,
                            "semantic_roles": 0.93
                        },
                        "confidence_score": 0.91,
                        "confidence_interval": [0.88, 0.94],
                        "uncertainty_factors": [],
                        "weights": {
                            "semantic": 0.45,
                            "structural": 0.4,
                            "constraint": 0.35,
                            "style": 0.2,
                            "intent": 0.3,
                            "complexity": -0.2
                        }
                    },
                    "attempts_used": 2,
                    "execution_trace": [],
                    "thinking_process": [],
                    "processing_time_ms": 1250
                }
            }
        }