import os
from typing import List, Dict, Any, Tuple
from .base import Tool
from models.hypothesis import PromptHypothesis
from models.validation import ValidationResult
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import re

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

class CohereValidator(Tool):
    """Validates prompt hypotheses using Cohere for semantic similarity"""
    
    def __init__(self):
        super().__init__(
            name="cohere_validator",
            description="Validates hypotheses using semantic similarity and constraint checking"
        )
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Cohere client"""
        if not COHERE_AVAILABLE:
            self.logger.warning("Cohere not available, using mock mode")
            self.client = None
            return
            
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            self.logger.warning("COHERE_API_KEY not set, using mock mode")
            self.client = None
            return
        
        try:
            # Try to use Cohere v2 client first
            if hasattr(cohere, 'ClientV2'):
                self.client = cohere.ClientV2(api_key=api_key)
                self.logger.info("Cohere ClientV2 initialized successfully")
            else:
                # Fallback to regular Client
                self.client = cohere.Client(api_key)
                self.logger.info("Cohere Client initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Cohere: {e}, using mock mode")
            self.client = None
    
    async def execute(self, input_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate each hypothesis"""
        hypothesis = input_data.get('hypothesis')
        original_output = input_data.get('original_output')
        
        self.logger.info(f"Validator called with {len(hypothesis) if isinstance(hypothesis, list) else 1} hypothesis(es)")
        
        if isinstance(hypothesis, list):
            # Validate multiple hypotheses
            results = []
            for hyp in hypothesis:
                result = await self._validate_single_hypothesis(hyp, original_output)
                results.append(result)
            return results
        else:
            # Validate single hypothesis
            result = await self._validate_single_hypothesis(hypothesis, original_output)
            return [result]
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _validate_single_hypothesis(
        self, 
        hypothesis: PromptHypothesis,
        original_output: str
    ) -> ValidationResult:
        """Validate a single hypothesis"""
        
        self.logger.info(f"Validating hypothesis: '{hypothesis.prompt[:50]}...' (confidence: {hypothesis.confidence})")
        
        # Generate output from hypothesis (simulated)
        simulated_output = await self._simulate_output_from_prompt(hypothesis.prompt)
        
        # Calculate semantic similarity
        semantic_similarity = await self._calculate_semantic_similarity(
            original_output, 
            simulated_output
        )
        
        # Check structural match
        structural_match = self._calculate_structural_match(
            original_output,
            simulated_output
        )
        
        # Check constraint satisfaction
        constraint_satisfaction = self._check_constraint_satisfaction(
            hypothesis,
            original_output
        )
        
        # Identify missing and extra elements
        missing_elements, extra_elements = self._identify_differences(
            hypothesis,
            original_output
        )
        
        # Calculate overall match score
        match_score = self._calculate_match_score(
            semantic_similarity,
            structural_match,
            constraint_satisfaction
        )
        
        # Calculate confidence calibration
        confidence_calibration = self._calculate_confidence_calibration(
            hypothesis.confidence,
            match_score
        )
        
        return ValidationResult(
            hypothesis=hypothesis.prompt,
            match_score=match_score,
            semantic_similarity=semantic_similarity,
            structural_match=structural_match,
            constraint_satisfaction=constraint_satisfaction,
            missing_elements=missing_elements,
            extra_elements=extra_elements,
            confidence_calibration=confidence_calibration
        )
    
    async def _simulate_output_from_prompt(self, prompt: str) -> str:
        """Simulate what output would be generated from the prompt"""
        if self.client is None:
            # Mock simulation for testing
            return self._mock_simulate_output(prompt)
            
        try:
            # Check if using ClientV2 or legacy Client
            if hasattr(cohere, 'ClientV2') and isinstance(self.client, cohere.ClientV2):
                # ClientV2 API
                response = self.client.chat(
                    model="command-r-plus",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.3
                )
                # Handle different response formats
                if hasattr(response, 'message') and hasattr(response.message, 'content'):
                    if isinstance(response.message.content, list):
                        return response.message.content[0].text.strip()
                    else:
                        return response.message.content.strip()
                elif hasattr(response, 'text'):
                    return response.text.strip()
                else:
                    return str(response).strip()
            else:
                # Legacy Client API
                response = self.client.generate(
                    model="command",
                    prompt=prompt,
                    max_tokens=300,
                    temperature=0.3
                )
                return response.generations[0].text.strip()
        except Exception as e:
            self.logger.error(f"Failed to simulate output: {str(e)}")
            return self._mock_simulate_output(prompt)
    
    def _mock_simulate_output(self, prompt: str) -> str:
        """Mock output simulation for testing"""
        prompt_lower = prompt.lower()
        if "list" in prompt_lower and "5" in prompt_lower:
            return "1. First item\n2. Second item\n3. Third item\n4. Fourth item\n5. Fifth item"
        elif "haiku" in prompt_lower:
            return "Lines of code flow\nBugs hidden in the shadows\nCoffee saves the day"
        else:
            return "This is a simulated output based on the prompt."
    
    async def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using Cohere embeddings"""
        if self.client is None:
            # Use simple similarity for mock mode
            return self._simple_text_similarity(text1, text2)
            
        try:
            # Check if using ClientV2 or legacy Client
            if hasattr(cohere, 'ClientV2') and isinstance(self.client, cohere.ClientV2):
                # ClientV2 API
                response = self.client.embed(
                    texts=[text1[:500], text2[:500]],  # Limit length
                    model='embed-english-v3.0',
                    input_type='search_document',
                    embedding_types=['float']
                )
                
                # Handle ClientV2 response format
                if hasattr(response, 'embeddings'):
                    embeddings = response.embeddings
                    if hasattr(embeddings, 'float') and len(embeddings.float) >= 2:
                        embedding1 = np.array(embeddings.float[0])
                        embedding2 = np.array(embeddings.float[1])
                    else:
                        raise ValueError("Unexpected embeddings format")
                    
                    # Calculate cosine similarity
                    cosine_sim = np.dot(embedding1, embedding2) / (
                        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
                    )
                    
                    # Normalize to 0-1 range
                    return float((cosine_sim + 1) / 2)
            else:
                # Legacy Client API
                response = self.client.embed(
                    texts=[text1[:500], text2[:500]],  # Limit length
                    model='embed-english-v2.0'  # Legacy model
                )
                
                # Legacy response is just a list of embeddings
                if hasattr(response, 'embeddings') and len(response.embeddings) >= 2:
                    embedding1 = np.array(response.embeddings[0])
                    embedding2 = np.array(response.embeddings[1])
                    
                    # Calculate cosine similarity
                    cosine_sim = np.dot(embedding1, embedding2) / (
                        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
                    )
                    
                    # Normalize to 0-1 range
                    return float((cosine_sim + 1) / 2)
                else:
                    self.logger.warning("Could not get embeddings, falling back to simple similarity")
                    return self._simple_text_similarity(text1, text2)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate semantic similarity: {str(e)}")
        
        # Fallback to simple text similarity
        return self._simple_text_similarity(text1, text2)
    
    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity as fallback"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_structural_match(self, original: str, simulated: str) -> float:
        """Calculate how well the structure matches"""
        features_original = self._extract_structural_features(original)
        features_simulated = self._extract_structural_features(simulated)
        
        matches = 0
        total = len(features_original)
        
        for feature, value in features_original.items():
            if feature in features_simulated:
                if isinstance(value, bool):
                    if value == features_simulated[feature]:
                        matches += 1
                elif isinstance(value, (int, float)):
                    # Allow some tolerance for numeric values
                    if abs(value - features_simulated[feature]) / max(value, 1) < 0.2:
                        matches += 1
        
        return matches / max(total, 1)
    
    def _extract_structural_features(self, text: str) -> Dict[str, Any]:
        """Extract basic structural features for comparison"""
        features = {}
        
        # Lists
        features['has_numbered_list'] = bool(re.search(r'^\d+[\.\)]\s+', text, re.MULTILINE))
        features['has_bullet_list'] = bool(re.search(r'^[\*\-\+]\s+', text, re.MULTILINE))
        
        # Counts
        features['line_count'] = len([l for l in text.split('\n') if l.strip()])
        features['word_count'] = len(text.split())
        features['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
        
        # Special elements
        features['has_code'] = '```' in text or bool(re.search(r'^\s{4,}', text, re.MULTILINE))
        features['has_questions'] = '?' in text
        
        return features
    
    def _check_constraint_satisfaction(
        self, 
        hypothesis: PromptHypothesis,
        output: str
    ) -> float:
        """Check how well constraints are satisfied"""
        satisfied = 0
        total = 0
        
        # Check for number constraints in hypothesis
        numbers = re.findall(r'\b(\d+)\b', hypothesis.prompt)
        for num in numbers:
            total += 1
            num_int = int(num)
            
            # Check if output has this many items/elements
            list_items = re.findall(r'^\d+[\.\)]\s+', output, re.MULTILINE)
            if len(list_items) == num_int:
                satisfied += 1
            
            # Check word count constraints
            words = output.split()
            if abs(len(words) - num_int) < num_int * 0.2:  # 20% tolerance
                satisfied += 0.5
        
        # Check for format keywords
        format_keywords = ['list', 'bullet', 'number', 'step', 'paragraph']
        for keyword in format_keywords:
            if keyword in hypothesis.prompt.lower():
                total += 1
                if keyword == 'list' and ('•' in output or re.search(r'^\d+[\.\)]', output, re.MULTILINE)):
                    satisfied += 1
                elif keyword == 'bullet' and '•' in output:
                    satisfied += 1
                elif keyword == 'number' and re.search(r'^\d+[\.\)]', output, re.MULTILINE):
                    satisfied += 1
        
        return satisfied / max(total, 1)
    
    def _identify_differences(
        self, 
        hypothesis: PromptHypothesis,
        output: str
    ) -> Tuple[List[str], List[str]]:
        """Identify missing and extra elements"""
        missing_elements = []
        extra_elements = []
        
        # Check key elements from hypothesis
        for element in hypothesis.key_elements:
            if element.lower() not in output.lower():
                missing_elements.append(f"Expected element: {element}")
        
        # Check for unexpected patterns in output
        output_features = self._extract_structural_features(output)
        
        if output_features.get('has_code') and 'code' not in hypothesis.prompt.lower():
            extra_elements.append("Unexpected code blocks")
        
        if output_features.get('has_questions') and 'question' not in hypothesis.prompt.lower():
            extra_elements.append("Unexpected questions")
        
        return missing_elements, extra_elements
    
    def _calculate_match_score(
        self,
        semantic_similarity: float,
        structural_match: float,
        constraint_satisfaction: float
    ) -> float:
        """Calculate overall match score with weighted average"""
        # Weights: semantic is most important
        weights = {
            'semantic': 0.5,
            'structural': 0.3,
            'constraint': 0.2
        }
        
        score = (
            semantic_similarity * weights['semantic'] +
            structural_match * weights['structural'] +
            constraint_satisfaction * weights['constraint']
        )
        
        return min(score, 1.0)
    
    def _calculate_confidence_calibration(
        self,
        predicted_confidence: float,
        actual_match: float
    ) -> float:
        """Calculate how well calibrated the confidence prediction was"""
        # Perfect calibration means predicted confidence matches actual match
        difference = abs(predicted_confidence - actual_match)
        
        # Convert to 0-1 score where 1 is perfect calibration
        calibration_score = 1.0 - difference
        
        return max(0.0, calibration_score)