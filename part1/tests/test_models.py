"""Unit tests for Pydantic models"""
import pytest
from pydantic import ValidationError
from models import (
    AnalyzeRequest, AnalyzeResponse, ErrorResponse,
    AnalysisReport, Pattern, PromptHypothesis,
    ValidationResult, DetectionResult,
    ConfidenceLevel, PatternType
)

class TestAnalyzeRequest:
    def test_valid_request(self):
        """Test creating a valid analyze request"""
        request = AnalyzeRequest(
            output_text="This is a test output",
            context="testing",
            max_attempts=2
        )
        assert request.output_text == "This is a test output"
        assert request.context == "testing"
        assert request.max_attempts == 2
    
    def test_request_validation(self):
        """Test request validation"""
        # Too short output
        with pytest.raises(ValidationError):
            AnalyzeRequest(output_text="short")
        
        # Too long output
        with pytest.raises(ValidationError):
            AnalyzeRequest(output_text="x" * 10001)
        
        # Invalid max_attempts
        with pytest.raises(ValidationError):
            AnalyzeRequest(output_text="Valid text here", max_attempts=0)
        
        with pytest.raises(ValidationError):
            AnalyzeRequest(output_text="Valid text here", max_attempts=6)
    
    def test_whitespace_trimming(self):
        """Test that whitespace is trimmed"""
        request = AnalyzeRequest(output_text="  Text with spaces  ")
        assert request.output_text == "Text with spaces"

class TestPattern:
    def test_valid_pattern(self):
        """Test creating a valid pattern"""
        pattern = Pattern(
            type=PatternType.STRUCTURAL,
            description="Test pattern",
            confidence=0.85,
            evidence=["Evidence 1", "Evidence 2"]
        )
        assert pattern.type == PatternType.STRUCTURAL
        assert pattern.confidence == 0.85
        assert len(pattern.evidence) == 2
    
    def test_confidence_bounds(self):
        """Test confidence score bounds"""
        with pytest.raises(ValidationError):
            Pattern(
                type=PatternType.LINGUISTIC,
                description="Test",
                confidence=1.5,  # Too high
                evidence=["Test"]
            )
        
        with pytest.raises(ValidationError):
            Pattern(
                type=PatternType.LINGUISTIC,
                description="Test",
                confidence=-0.1,  # Too low
                evidence=["Test"]
            )

class TestPromptHypothesis:
    def test_valid_hypothesis(self):
        """Test creating a valid hypothesis"""
        hypothesis = PromptHypothesis(
            prompt="Test prompt",
            confidence=0.75,
            reasoning="Test reasoning",
            key_elements=["test", "prompt"],
            rank=1
        )
        assert hypothesis.prompt == "Test prompt"
        assert hypothesis.confidence == 0.75
        assert hypothesis.rank == 1
    
    def test_rank_validation(self):
        """Test rank must be positive"""
        with pytest.raises(ValidationError):
            PromptHypothesis(
                prompt="Test",
                confidence=0.5,
                reasoning="Test",
                key_elements=[],
                rank=0  # Invalid
            )

class TestValidationResult:
    def test_valid_validation(self):
        """Test creating a valid validation result"""
        result = ValidationResult(
            hypothesis="Test hypothesis",
            match_score=0.88,
            semantic_similarity=0.92,
            structural_match=0.85,
            constraint_satisfaction=0.90,
            missing_elements=["element1"],
            extra_elements=[],
            confidence_calibration=0.87
        )
        assert result.match_score == 0.88
        assert len(result.missing_elements) == 1
        assert len(result.extra_elements) == 0
    
    def test_score_bounds(self):
        """Test all scores are bounded 0-1"""
        with pytest.raises(ValidationError):
            ValidationResult(
                hypothesis="Test",
                match_score=1.1,  # Too high
                semantic_similarity=0.5,
                structural_match=0.5,
                constraint_satisfaction=0.5,
                missing_elements=[],
                extra_elements=[],
                confidence_calibration=0.5
            )

class TestConfidenceLevel:
    def test_confidence_levels(self):
        """Test confidence level enum"""
        assert ConfidenceLevel.VERY_HIGH.value == "very_high"
        assert ConfidenceLevel.LOW.value == "low"
        
        # Test all levels exist
        levels = [level.value for level in ConfidenceLevel]
        assert "very_high" in levels
        assert "high" in levels
        assert "medium" in levels
        assert "low" in levels
        assert "very_low" in levels

class TestErrorResponse:
    def test_error_response(self):
        """Test error response creation"""
        error = ErrorResponse(
            message="Test error",
            error_code="TEST_ERROR",
            details={"field": "value"}
        )
        assert error.success is False
        assert error.error_code == "TEST_ERROR"
        assert error.details["field"] == "value"