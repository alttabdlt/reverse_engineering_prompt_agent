from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class ConfidenceLevel(str, Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

class PatternType(str, Enum):
    STRUCTURAL = "structural"
    LINGUISTIC = "linguistic"
    CONSTRAINT = "constraint"
    META = "meta"

class BaseResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    message: Optional[str] = Field(None, description="Additional message or error description")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

class ErrorResponse(BaseResponse):
    success: bool = False
    error_code: str = Field(..., description="Error code for debugging")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")