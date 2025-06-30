"""Unit tests for the Prompt Detective Agent"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from agents.prompt_detective import PromptDetectiveAgent
from models import (
    AnalysisReport, Pattern, PromptHypothesis, HypothesisSet,
    ValidationResult, ConfidenceLevel, PatternType
)

class TestPromptDetectiveAgent:
    @pytest.fixture
    def agent(self):
        return PromptDetectiveAgent()
    
    @pytest.fixture
    def mock_analysis(self):
        """Mock analysis report"""
        return AnalysisReport(
            patterns=[
                Pattern(
                    type=PatternType.STRUCTURAL,
                    description="Numbered list",
                    confidence=0.9,
                    evidence=["5 items detected"]
                )
            ],
            structural_features={
                'has_numbered_list': True,
                'list_item_count': 5
            },
            linguistic_features={
                'word_count': 50,
                'formality_score': 0
            },
            constraints_detected=["Exactly 5 items"],
            overall_confidence=ConfidenceLevel.HIGH
        )
    
    @pytest.fixture
    def mock_hypotheses(self):
        """Mock hypothesis set"""
        return HypothesisSet(
            hypotheses=[
                PromptHypothesis(
                    prompt="List 5 benefits of exercise",
                    confidence=0.85,
                    reasoning="Numbered list with 5 items about benefits",
                    key_elements=["List", "5", "benefits", "exercise"],
                    rank=1
                ),
                PromptHypothesis(
                    prompt="Give me 5 advantages of working out",
                    confidence=0.75,
                    reasoning="Alternative phrasing",
                    key_elements=["5", "advantages", "working out"],
                    rank=2
                )
            ],
            analysis_context={'patterns_found': 1},
            iteration=1
        )
    
    @pytest.fixture
    def mock_validation(self):
        """Mock validation results"""
        return [
            ValidationResult(
                hypothesis="List 5 benefits of exercise",
                match_score=0.88,
                semantic_similarity=0.92,
                structural_match=0.95,
                constraint_satisfaction=0.85,
                missing_elements=[],
                extra_elements=[],
                confidence_calibration=0.86
            )
        ]
    
    @pytest.mark.asyncio
    async def test_detect_prompt_success(self, agent, mock_analysis, mock_hypotheses, mock_validation):
        """Test successful prompt detection"""
        # Mock the tools
        agent.pattern_analyzer.execute = AsyncMock(return_value=mock_analysis)
        agent.hypothesis_generator.execute = AsyncMock(return_value=mock_hypotheses)
        agent.validator.execute = AsyncMock(return_value=mock_validation)
        
        # Test detection
        output_text = "1. Improves health\n2. Reduces stress\n3. Increases energy\n4. Better sleep\n5. Weight control"
        result = await agent.detect_prompt(output_text, max_attempts=1)
        
        # Verify result
        assert result.best_hypothesis.prompt == "List 5 benefits of exercise"
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.attempts_used == 1
        assert len(result.all_hypotheses) == 2
        assert len(result.validation_results) == 1
    
    @pytest.mark.asyncio
    async def test_multi_pass_refinement(self, agent, mock_analysis):
        """Test multi-pass refinement process"""
        # Mock tools with improving scores
        agent.pattern_analyzer.execute = AsyncMock(return_value=mock_analysis)
        
        # First pass - low score
        agent.hypothesis_generator.execute = AsyncMock(side_effect=[
            HypothesisSet(
                hypotheses=[
                    PromptHypothesis(
                        prompt="Write about exercise",
                        confidence=0.6,
                        reasoning="Generic match",
                        key_elements=["exercise"],
                        rank=1
                    )
                ],
                analysis_context={},
                iteration=1
            ),
            # Second pass - better score
            HypothesisSet(
                hypotheses=[
                    PromptHypothesis(
                        prompt="List 5 benefits of exercise",
                        confidence=0.85,
                        reasoning="Refined match",
                        key_elements=["List", "5", "benefits"],
                        rank=1
                    )
                ],
                analysis_context={},
                iteration=2
            )
        ])
        
        agent.validator.execute = AsyncMock(side_effect=[
            # First validation - low score
            [ValidationResult(
                hypothesis="Write about exercise",
                match_score=0.65,
                semantic_similarity=0.7,
                structural_match=0.6,
                constraint_satisfaction=0.5,
                missing_elements=["list format", "specific count"],
                extra_elements=[],
                confidence_calibration=0.6
            )],
            # Second validation - high score
            [ValidationResult(
                hypothesis="List 5 benefits of exercise",
                match_score=0.88,
                semantic_similarity=0.92,
                structural_match=0.95,
                constraint_satisfaction=0.85,
                missing_elements=[],
                extra_elements=[],
                confidence_calibration=0.86
            )]
        ])
        
        output_text = "1. Health\n2. Stress\n3. Energy\n4. Sleep\n5. Weight"
        result = await agent.detect_prompt(output_text, max_attempts=3)
        
        # Should have done 2 passes
        assert result.attempts_used == 2
        assert result.best_hypothesis.prompt == "List 5 benefits of exercise"
        assert len(result.all_hypotheses) == 2
    
    @pytest.mark.asyncio
    async def test_confidence_determination(self, agent):
        """Test confidence level determination"""
        # Test different match scores
        assert agent._determine_confidence(0.95) == ConfidenceLevel.VERY_HIGH
        assert agent._determine_confidence(0.85) == ConfidenceLevel.HIGH
        assert agent._determine_confidence(0.65) == ConfidenceLevel.MEDIUM
        assert agent._determine_confidence(0.45) == ConfidenceLevel.LOW
        assert agent._determine_confidence(0.25) == ConfidenceLevel.VERY_LOW
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling in detection"""
        # Mock pattern analyzer to raise error
        agent.pattern_analyzer.execute = AsyncMock(side_effect=Exception("Analysis failed"))
        
        with pytest.raises(Exception) as exc_info:
            await agent.detect_prompt("Test output")
        
        assert "Analysis failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_execution_trace(self, agent, mock_analysis, mock_hypotheses, mock_validation):
        """Test execution trace is properly recorded"""
        agent.pattern_analyzer.execute = AsyncMock(return_value=mock_analysis)
        agent.hypothesis_generator.execute = AsyncMock(return_value=mock_hypotheses)
        agent.validator.execute = AsyncMock(return_value=mock_validation)
        
        result = await agent.detect_prompt("Test output", max_attempts=1)
        
        # Check execution trace
        assert len(result.execution_trace) >= 1
        assert result.execution_trace[0]['pass'] == 1
        assert result.execution_trace[0]['action'] == 'initial_analysis'
        assert 'timestamp' in result.execution_trace[0]
    
    @pytest.mark.asyncio
    async def test_processing_time(self, agent, mock_analysis, mock_hypotheses, mock_validation):
        """Test processing time is recorded"""
        agent.pattern_analyzer.execute = AsyncMock(return_value=mock_analysis)
        agent.hypothesis_generator.execute = AsyncMock(return_value=mock_hypotheses)
        agent.validator.execute = AsyncMock(return_value=mock_validation)
        
        result = await agent.detect_prompt("Test output")
        
        assert result.processing_time_ms > 0
        assert isinstance(result.processing_time_ms, int)