from pydantic import BaseModel, Field, field_validator
from typing import Optional

class AnalyzeRequest(BaseModel):
    output_text: str = Field(
        ..., 
        description="The AI-generated output to analyze",
        min_length=10,
        max_length=10000
    )
    context: Optional[str] = Field(
        None,
        description="Additional context about the output (e.g., model used, domain)"
    )
    max_attempts: int = Field(
        5,
        description="Maximum number of refinement attempts (up to 5 passes)",
        ge=1,
        le=5
    )
    
    @field_validator('output_text')
    def validate_output_text(cls, v):
        if not v.strip():
            raise ValueError("Output text cannot be empty or just whitespace")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "output_text": "Here are 5 benefits of regular exercise:\n1. Improves cardiovascular health\n2. Boosts mood and mental health\n3. Helps with weight management\n4. Increases energy levels\n5. Promotes better sleep",
                "context": "Health and wellness domain",
                "max_attempts": 5
            }
        }