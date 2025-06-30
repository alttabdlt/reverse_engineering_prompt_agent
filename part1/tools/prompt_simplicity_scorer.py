"""Prompt Simplicity Scorer - Applies Occam's Razor to prompt reconstruction"""

import re
from typing import Dict, List, Tuple
from .base import Tool
import math

class PromptSimplicityScorer(Tool):
    """Scores prompts based on simplicity and likelihood principles"""
    
    def __init__(self):
        super().__init__(
            name="prompt_simplicity_scorer",
            description="Evaluates prompt simplicity using Occam's Razor principles"
        )
        
        # Common prompt patterns (from most to least common)
        self.common_patterns = [
            # Simple questions
            (r'^(what|how|why|when|where|who)\s+.+\??$', 0.9, "simple_question"),
            (r'^(explain|describe|define)\s+.+$', 0.85, "explanation_request"),
            
            # Basic instructions
            (r'^(write|create|generate)\s+(?:a\s+)?(\w+)(?:\s+about\s+.+)?$', 0.8, "creative_request"),
            (r'^(list|enumerate|name)\s+.+$', 0.75, "listing_request"),
            (r'^(summarize|analyze|compare)\s+.+$', 0.7, "analysis_request"),
            
            # More complex
            (r'^.+\s+in\s+\d+\s+words?$', 0.5, "word_count_constraint"),
            (r'^.+\s+(?:using|with)\s+.+\s+format$', 0.4, "format_constraint"),
            (r'^.+\s+(?:include|including)\s+.+$', 0.4, "inclusion_constraint"),
            
            # Very specific
            (r'\d+\s+(?:bullet\s+)?points?', 0.3, "bullet_count"),
            (r'exactly\s+\d+', 0.2, "exact_count"),
            (r'(?:first|then|finally)', 0.3, "sequential_steps"),
        ]
        
        # Complexity indicators
        self.complexity_indicators = {
            'word_count': lambda p: len(p.split()),
            'sentence_count': lambda p: len(re.split(r'[.!?]+', p)),
            'constraint_count': lambda p: self._count_constraints(p),
            'specificity_score': lambda p: self._calculate_specificity(p),
        }
        
    async def execute(self, input_data: Dict[str, any]) -> Dict[str, float]:
        """Score a prompt for simplicity"""
        prompt = input_data.get('prompt', '')
        
        # Calculate various scores
        pattern_score = self._calculate_pattern_score(prompt)
        length_score = self._calculate_length_score(prompt)
        constraint_score = self._calculate_constraint_score(prompt)
        specificity_score = self._calculate_specificity(prompt)
        
        # Combine scores with weights
        weights = {
            'pattern': 0.3,
            'length': 0.25,
            'constraints': 0.25,
            'specificity': 0.2
        }
        
        simplicity_score = (
            pattern_score * weights['pattern'] +
            length_score * weights['length'] +
            constraint_score * weights['constraints'] +
            specificity_score * weights['specificity']
        )
        
        # Calculate prompt likelihood (prior probability)
        likelihood = self._calculate_prompt_likelihood(prompt)
        
        return {
            'simplicity_score': simplicity_score,
            'pattern_score': pattern_score,
            'length_score': length_score, 
            'constraint_score': constraint_score,
            'specificity_score': specificity_score,
            'likelihood': likelihood,
            'complexity_breakdown': self._get_complexity_breakdown(prompt),
            'detected_patterns': self._get_detected_patterns(prompt)
        }
    
    def _calculate_pattern_score(self, prompt: str) -> float:
        """Score based on how well prompt matches common patterns"""
        prompt_lower = prompt.lower().strip()
        
        for pattern, base_score, pattern_type in self.common_patterns:
            if re.match(pattern, prompt_lower):
                # Adjust score based on additional complexity
                complexity_penalty = self._count_constraints(prompt) * 0.1
                return max(0.1, base_score - complexity_penalty)
        
        # No common pattern matched
        return 0.3
    
    def _calculate_length_score(self, prompt: str) -> float:
        """Score based on prompt length (shorter is simpler)"""
        word_count = len(prompt.split())
        
        if word_count <= 5:
            return 1.0
        elif word_count <= 10:
            return 0.8
        elif word_count <= 20:
            return 0.6
        elif word_count <= 40:
            return 0.4
        else:
            # Logarithmic decay for very long prompts
            return max(0.1, 0.4 - math.log10(word_count / 40) * 0.2)
    
    def _calculate_constraint_score(self, prompt: str) -> float:
        """Score based on number of constraints (fewer is simpler)"""
        constraint_count = self._count_constraints(prompt)
        
        if constraint_count == 0:
            return 1.0
        elif constraint_count == 1:
            return 0.7
        elif constraint_count == 2:
            return 0.5
        elif constraint_count == 3:
            return 0.3
        else:
            return max(0.1, 0.3 - constraint_count * 0.05)
    
    def _count_constraints(self, prompt: str) -> int:
        """Count number of constraints in prompt"""
        constraints = 0
        prompt_lower = prompt.lower()
        
        # Specific constraints to look for
        constraint_patterns = [
            r'\d+\s+words?',  # Word count
            r'\d+\s+(?:bullet|point|item)s?',  # Item count
            r'(?:using|in|with)\s+.+\s+(?:format|style)',  # Format specification
            r'include\s+(?:the\s+)?(?:following|these)',  # Inclusion requirements
            r'(?:exactly|precisely|specifically)',  # Exactness requirements
            r'(?:must|should)\s+(?:have|contain|include)',  # Requirements
            r'(?:focus\s+on|emphasize|highlight)',  # Focus constraints
            r'(?:avoid|don\'t|do\s+not)',  # Exclusion constraints
        ]
        
        for pattern in constraint_patterns:
            if re.search(pattern, prompt_lower):
                constraints += 1
        
        return constraints
    
    def _calculate_specificity(self, prompt: str) -> float:
        """Calculate how specific vs general the prompt is"""
        specificity_indicators = [
            (r'\b(?:exactly|precisely|specifically)\b', -0.2),
            (r'\b\d+\b', -0.15),  # Specific numbers
            (r'\b(?:must|should|need\s+to)\b', -0.1),
            (r'"[^"]+"', -0.15),  # Quoted text
            (r'\b(?:format|structure|style)\b', -0.1),
            (r'\b(?:general|brief|simple|basic)\b', 0.1),
            (r'\b(?:explain|describe|what\s+is)\b', 0.05),
        ]
        
        score = 0.7  # Start with moderate specificity
        
        for pattern, adjustment in specificity_indicators:
            if re.search(pattern, prompt.lower()):
                score += adjustment
        
        return max(0.1, min(1.0, score))
    
    def _calculate_prompt_likelihood(self, prompt: str) -> float:
        """Calculate prior probability of this being a real user prompt"""
        # Based on empirical observations of common prompts
        word_count = len(prompt.split())
        
        # Most prompts are 3-15 words
        if 3 <= word_count <= 15:
            base_likelihood = 0.7
        elif word_count < 3:
            base_likelihood = 0.3  # Too short
        elif word_count <= 30:
            base_likelihood = 0.5
        else:
            base_likelihood = 0.2  # Very long prompts are rare
        
        # Adjust based on pattern matching
        prompt_lower = prompt.lower()
        
        # Common starting words increase likelihood
        common_starts = ['explain', 'write', 'what', 'how', 'create', 'list', 'describe']
        if any(prompt_lower.startswith(word) for word in common_starts):
            base_likelihood += 0.1
        
        # Too many constraints decrease likelihood
        if self._count_constraints(prompt) > 3:
            base_likelihood -= 0.2
        
        return max(0.1, min(0.9, base_likelihood))
    
    def _get_complexity_breakdown(self, prompt: str) -> Dict[str, any]:
        """Get detailed complexity analysis"""
        return {
            'word_count': len(prompt.split()),
            'character_count': len(prompt),
            'sentence_count': len(re.split(r'[.!?]+', prompt.strip())) - 1,
            'constraint_count': self._count_constraints(prompt),
            'has_specific_numbers': bool(re.search(r'\b\d+\b', prompt)),
            'has_format_requirements': bool(re.search(r'(?:format|style|structure)', prompt.lower())),
            'has_length_requirements': bool(re.search(r'\d+\s*(?:words?|sentences?|paragraphs?)', prompt.lower())),
        }
    
    def _get_detected_patterns(self, prompt: str) -> List[str]:
        """Get list of detected patterns in prompt"""
        detected = []
        prompt_lower = prompt.lower().strip()
        
        for pattern, _, pattern_type in self.common_patterns:
            if re.match(pattern, prompt_lower):
                detected.append(pattern_type)
        
        return detected