"""Progressive Hint Engine - Provides calibrated hints for reverse prompt engineering"""

from typing import Dict, Any, Optional, List
from .base import Tool
import re

class ProgressiveHintEngine(Tool):
    """Provides progressive hints based on iteration number, similar to Hangman"""
    
    def __init__(self):
        super().__init__(
            name="progressive_hint_engine",
            description="Provides calibrated hints to guide prompt reconstruction"
        )
        
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hints based on iteration and original prompt"""
        iteration = input_data.get('iteration', 1)
        original_prompt = input_data.get('original_prompt', '')
        previous_attempts = input_data.get('previous_attempts', [])
        
        # Generate appropriate hints for this iteration
        hints = self._generate_hints(original_prompt, iteration, previous_attempts)
        
        return {
            'hints': hints,
            'iteration': iteration,
            'hint_level': self._get_hint_level(iteration)
        }
    
    def _generate_hints(self, original_prompt: str, iteration: int, previous_attempts: List[str]) -> Dict[str, Any]:
        """Generate hints appropriate for the current iteration"""
        hints = {}
        
        # Always provide some context about previous attempts
        if previous_attempts:
            hints['feedback'] = self._analyze_previous_attempts(original_prompt, previous_attempts)
        
        # Progressive hint disclosure
        if iteration >= 1:
            # Pass 1: No hints (let system try on its own)
            hints['hint_level'] = 'none'
            hints['guidance'] = 'Try to find the simplest prompt that could generate this output'
        
        if iteration >= 2:
            # Pass 2: Length category
            word_count = len(original_prompt.split())
            hints['hint_level'] = 'length_category'
            hints['length_category'] = self._categorize_length(word_count)
            hints['guidance'] = f'The prompt is {hints["length_category"]} in length'
        
        if iteration >= 3:
            # Pass 3: More specific length and type
            word_count = len(original_prompt.split())
            hints['hint_level'] = 'length_and_type'
            hints['word_count_range'] = self._get_word_count_range(word_count)
            hints['prompt_type'] = self._categorize_prompt_type(original_prompt)
            hints['guidance'] = f'The prompt is {hints["word_count_range"]} words and is a {hints["prompt_type"]}'
        
        if iteration >= 4:
            # Pass 4: Starting word and structure hints
            hints['hint_level'] = 'structure'
            hints['starts_with'] = original_prompt.split()[0] if original_prompt else ''
            hints['sentence_type'] = self._get_sentence_type(original_prompt)
            hints['has_constraints'] = self._has_constraints(original_prompt)
            
            # Masked preview (like Hangman)
            hints['masked_preview'] = self._create_masked_preview(original_prompt, reveal_percent=0.2)
            hints['guidance'] = f'The prompt starts with "{hints["starts_with"]}" and is a {hints["sentence_type"]}'
        
        if iteration >= 5:
            # Pass 5: More revealing hints
            hints['hint_level'] = 'detailed'
            hints['exact_word_count'] = len(original_prompt.split())
            hints['key_terms'] = self._extract_key_terms(original_prompt)
            hints['masked_preview'] = self._create_masked_preview(original_prompt, reveal_percent=0.4)
            
            # Provide structural pattern
            hints['pattern'] = self._extract_pattern(original_prompt)
            hints['guidance'] = 'Here are significant hints about the prompt structure'
        
        return hints
    
    def _categorize_length(self, word_count: int) -> str:
        """Categorize prompt length"""
        if word_count <= 5:
            return "very short"
        elif word_count <= 10:
            return "short"
        elif word_count <= 20:
            return "medium"
        elif word_count <= 40:
            return "long"
        else:
            return "very long"
    
    def _get_word_count_range(self, word_count: int) -> str:
        """Get word count range"""
        if word_count <= 3:
            return "1-3"
        elif word_count <= 5:
            return "4-5"
        elif word_count <= 10:
            return "6-10"
        elif word_count <= 15:
            return "11-15"
        elif word_count <= 25:
            return "16-25"
        else:
            return f"26-{word_count+5}"
    
    def _categorize_prompt_type(self, prompt: str) -> str:
        """Categorize the type of prompt"""
        prompt_lower = prompt.lower()
        
        # Check for question
        if prompt.endswith('?') or prompt_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who')):
            return "question"
        
        # Check for instruction
        elif prompt_lower.startswith(('write', 'create', 'generate', 'make', 'produce')):
            return "creative instruction"
        
        # Check for explanation request
        elif prompt_lower.startswith(('explain', 'describe', 'define', 'elaborate')):
            return "explanation request"
        
        # Check for analysis
        elif prompt_lower.startswith(('analyze', 'compare', 'evaluate', 'assess')):
            return "analysis request"
        
        # Check for listing
        elif prompt_lower.startswith(('list', 'enumerate', 'name', 'provide')):
            return "listing request"
        
        else:
            return "general instruction"
    
    def _get_sentence_type(self, prompt: str) -> str:
        """Determine sentence type"""
        if prompt.endswith('?'):
            return "question"
        elif prompt.endswith('!'):
            return "exclamation"
        elif any(prompt.lower().startswith(word) for word in ['please', 'could', 'would']):
            return "polite request"
        else:
            return "statement/instruction"
    
    def _has_constraints(self, prompt: str) -> bool:
        """Check if prompt has specific constraints"""
        constraint_patterns = [
            r'\d+\s+words?',
            r'\d+\s+(?:bullet|point|item)s?',
            r'(?:using|in|with)\s+.+\s+(?:format|style)',
            r'include\s+(?:the\s+)?(?:following|these)',
            r'(?:exactly|precisely|specifically)',
        ]
        
        prompt_lower = prompt.lower()
        return any(re.search(pattern, prompt_lower) for pattern in constraint_patterns)
    
    def _create_masked_preview(self, prompt: str, reveal_percent: float = 0.3) -> str:
        """Create a Hangman-style masked preview of the prompt"""
        if not prompt:
            return ""
        
        words = prompt.split()
        total_chars = sum(len(word) for word in words)
        chars_to_reveal = int(total_chars * reveal_percent)
        
        # Always reveal first word and punctuation
        masked_words = []
        chars_revealed = 0
        
        for i, word in enumerate(words):
            if i == 0 or chars_revealed < chars_to_reveal:
                # Reveal this word
                masked_words.append(word)
                chars_revealed += len(word)
            else:
                # Mask this word but keep same length
                masked = ''.join('_' if c.isalnum() else c for c in word)
                masked_words.append(masked)
        
        return ' '.join(masked_words)
    
    def _extract_key_terms(self, prompt: str) -> List[str]:
        """Extract key terms from prompt (nouns, verbs)"""
        # Simple extraction - in production, use NLP library
        words = prompt.lower().split()
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        return key_terms[:5]  # Return top 5 key terms
    
    def _extract_pattern(self, prompt: str) -> str:
        """Extract the pattern of the prompt"""
        words = prompt.split()
        
        # Create pattern representation
        pattern_parts = []
        for word in words:
            if word.lower() in ['the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of']:
                pattern_parts.append(word.lower())
            elif word[0].isupper():
                pattern_parts.append('[Noun]')
            elif word.endswith('ing') or word.endswith('ed'):
                pattern_parts.append('[Verb]')
            else:
                pattern_parts.append('[Word]')
        
        return ' '.join(pattern_parts)
    
    def _analyze_previous_attempts(self, original: str, attempts: List[str]) -> Dict[str, Any]:
        """Analyze previous attempts to provide feedback"""
        feedback = {
            'attempts_made': len(attempts),
            'getting_closer': False,
            'common_mistakes': []
        }
        
        if not attempts:
            return feedback
        
        # Check if attempts are getting closer in length
        original_length = len(original.split())
        last_attempt_length = len(attempts[-1].split())
        
        if len(attempts) > 1:
            prev_attempt_length = len(attempts[-2].split())
            length_diff_current = abs(original_length - last_attempt_length)
            length_diff_prev = abs(original_length - prev_attempt_length)
            feedback['getting_closer'] = length_diff_current < length_diff_prev
        
        # Identify common mistakes
        for attempt in attempts:
            if len(attempt.split()) > original_length * 2:
                feedback['common_mistakes'].append('over_specification')
            if 'exactly' in attempt.lower() and 'exactly' not in original.lower():
                feedback['common_mistakes'].append('unnecessary_precision')
            if attempt.count(',') > original.count(','):
                feedback['common_mistakes'].append('too_many_constraints')
        
        return feedback
    
    def _get_hint_level(self, iteration: int) -> str:
        """Get descriptive hint level"""
        levels = {
            1: "no_hints",
            2: "minimal", 
            3: "moderate",
            4: "substantial",
            5: "revealing"
        }
        return levels.get(iteration, "maximum")