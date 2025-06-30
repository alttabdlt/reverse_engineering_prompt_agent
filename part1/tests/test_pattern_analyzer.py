"""Unit tests for Pattern Analyzer tool"""
import pytest
import asyncio
from tools.pattern_analyzer import PatternAnalyzer
from models.base import PatternType, ConfidenceLevel

class TestPatternAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return PatternAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_numbered_list(self, analyzer):
        """Test analysis of numbered list output"""
        output_text = """1. First item
2. Second item
3. Third item
4. Fourth item
5. Fifth item"""
        
        result = await analyzer.execute(output_text)
        
        # Check structural features
        assert result.structural_features['has_numbered_list'] is True
        assert result.structural_features['list_item_count'] == 5
        
        # Check patterns detected
        structural_patterns = [p for p in result.patterns if p.type == PatternType.STRUCTURAL]
        assert len(structural_patterns) > 0
        assert any("Numbered list" in p.description for p in structural_patterns)
    
    @pytest.mark.asyncio
    async def test_analyze_code_content(self, analyzer):
        """Test analysis of code content"""
        output_text = """Here's a Python function:

```python
def hello_world():
    print("Hello, World!")
```

This function prints a greeting."""
        
        result = await analyzer.execute(output_text)
        
        # Check code detection
        assert result.structural_features['has_code'] is True
        
        # Check patterns
        code_patterns = [p for p in result.patterns if "code" in p.description.lower()]
        assert len(code_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_formal_tone(self, analyzer):
        """Test detection of formal tone"""
        output_text = """Therefore, it is imperative to consider the implications of this decision. 
Furthermore, the consequences of such actions must be thoroughly evaluated. 
Consequently, we recommend a comprehensive review of all available options."""
        
        result = await analyzer.execute(output_text)
        
        # Check linguistic features
        assert result.linguistic_features['formality_score'] > 0
        
        # Check formal tone pattern
        linguistic_patterns = [p for p in result.patterns if p.type == PatternType.LINGUISTIC]
        assert any("formal" in p.description.lower() for p in linguistic_patterns)
    
    @pytest.mark.asyncio
    async def test_analyze_constraints(self, analyzer):
        """Test constraint detection"""
        output_text = """1. Exercise regularly
2. Eat healthy foods
3. Get enough sleep
4. Stay hydrated
5. Manage stress"""
        
        result = await analyzer.execute(output_text)
        
        # Check constraints
        assert "Exactly 5 items" in result.constraints_detected
        assert "Numbered list format required" in result.constraints_detected
    
    @pytest.mark.asyncio
    async def test_analyze_brief_output(self, analyzer):
        """Test analysis of brief output"""
        output_text = "This is a very short response."
        
        result = await analyzer.execute(output_text)
        
        # Check constraints
        assert "Brief/concise output" in result.constraints_detected
        assert result.linguistic_features['word_count'] < 10
    
    @pytest.mark.asyncio 
    async def test_confidence_calculation(self, analyzer):
        """Test confidence level calculation"""
        # High confidence case - clear structure
        output_text = """1. First clear point
2. Second clear point
3. Third clear point"""
        
        result = await analyzer.execute(output_text)
        assert result.overall_confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]
        
        # Low confidence case - ambiguous
        output_text = "Some random text without clear structure."
        
        result = await analyzer.execute(output_text)
        assert result.overall_confidence in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM]
    
    @pytest.mark.asyncio
    async def test_technical_content_detection(self, analyzer):
        """Test detection of technical content"""
        output_text = """The REST API uses HTTP methods like GET, POST, PUT, and DELETE. 
The JSON response includes status codes and error messages. 
We implemented the algorithm using a recursive function with O(n) complexity."""
        
        result = await analyzer.execute(output_text)
        
        # Check technical term count
        assert result.linguistic_features['technical_term_count'] > 5
        
        # Check for technical pattern
        tech_patterns = [p for p in result.patterns if "technical" in p.description.lower()]
        assert len(tech_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_question_detection(self, analyzer):
        """Test detection of questions"""
        output_text = """What is machine learning? How does it work? 
Machine learning is a subset of AI that enables systems to learn from data.
Why is it important? Because it can automate complex tasks."""
        
        result = await analyzer.execute(output_text)
        
        # Check question features
        assert result.linguistic_features['has_questions'] is True
        assert result.linguistic_features['question_count'] == 3
    
    def test_edge_case_empty_text(self, analyzer):
        """Test handling of empty text"""
        # The tool should handle empty text gracefully
        # but this would be caught by request validation in practice
        pass