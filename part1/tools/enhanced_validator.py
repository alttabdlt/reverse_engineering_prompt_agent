"""Enhanced validator with sophisticated scoring system"""

import os
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .base import Tool
from models.hypothesis import PromptHypothesis
from models.validation import ValidationResult
from models.responses import EnhancedScoring, PromptComparison
import statistics
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

class EnhancedValidator(Tool):
    """Enhanced validator with sophisticated scoring and comparison features"""
    
    def __init__(self):
        super().__init__(
            name="enhanced_validator",
            description="Advanced validation with multi-dimensional scoring"
        )
        self._initialize_client()
        
        # Dynamic weight profiles for different prompt types
        self.weight_profiles = {
            "technical": {
                "semantic": 0.4,
                "structural": 0.35,
                "constraint": 0.25,
                "style": 0.15,
                "intent": 0.35,
                "complexity": -0.15
            },
            "creative": {
                "semantic": 0.5,
                "structural": 0.2,
                "constraint": 0.15,
                "style": 0.35,
                "intent": 0.4,
                "complexity": -0.1
            },
            "instructional": {
                "semantic": 0.45,
                "structural": 0.4,
                "constraint": 0.35,
                "style": 0.2,
                "intent": 0.3,
                "complexity": -0.2
            },
            "default": {
                "semantic": 0.45,
                "structural": 0.3,
                "constraint": 0.25,
                "style": 0.25,
                "intent": 0.35,
                "complexity": -0.15
            }
        }
    
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
                self.logger.info("Cohere ClientV2 initialized for enhanced validation")
            else:
                # Fallback to regular Client
                self.client = cohere.Client(
                    api_key,
                    timeout=30  # 30 second timeout instead of default 120
                )
                self.logger.info("Cohere Client initialized for enhanced validation")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Cohere: {e}, using mock mode")
            self.client = None
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced validation with comparison and scoring"""
        hypothesis = input_data.get('hypothesis')
        original_output = input_data.get('original_output')
        
        if isinstance(hypothesis, PromptHypothesis):
            # Single hypothesis validation
            result = await self._validate_with_enhanced_scoring(hypothesis, original_output)
            comparison = await self._generate_comparison(hypothesis, original_output)
            
            return {
                "validation_result": result,
                "comparison": comparison,
                "enhanced_scoring": result.get("enhanced_scoring")
            }
        else:
            # Multiple hypotheses
            results = []
            for hyp in hypothesis:
                result = await self._validate_with_enhanced_scoring(hyp, original_output)
                results.append(result)
            
            # Generate comparison for best hypothesis
            if results:
                best_idx = np.argmax([r.match_score for r in results])
                comparison = await self._generate_comparison(hypothesis[best_idx], original_output)
            else:
                comparison = None
                
            return {
                "validation_results": results,
                "comparison": comparison,
                "enhanced_scoring": results[0].get("enhanced_scoring") if results else None
            }
    
    async def _validate_with_enhanced_scoring(
        self, 
        hypothesis: PromptHypothesis,
        original_output: str
    ) -> ValidationResult:
        """Validate with enhanced multi-dimensional scoring"""
        
        # Detect prompt type
        prompt_type = self._detect_prompt_type(hypothesis.prompt)
        weights = self.weight_profiles.get(prompt_type, self.weight_profiles["default"])
        
        # Generate simulated output
        simulated_output = await self._simulate_output_from_prompt(hypothesis.prompt)
        
        # Calculate all scoring dimensions
        scores = {}
        
        # 1. Semantic Similarity (enhanced)
        semantic_score, semantic_components = await self._calculate_enhanced_semantic_similarity(
            original_output, simulated_output
        )
        scores["semantic"] = semantic_score
        
        # 2. Structural Match (enhanced)
        structural_score, structural_components = self._calculate_enhanced_structural_match(
            original_output, simulated_output
        )
        scores["structural"] = structural_score
        
        # 3. Constraint Satisfaction (enhanced)
        constraint_score = self._calculate_enhanced_constraint_satisfaction(
            hypothesis, original_output
        )
        scores["constraint"] = constraint_score
        
        # 4. Style and Tone Match
        style_score = await self._calculate_style_match(
            original_output, simulated_output
        )
        scores["style"] = style_score
        
        # 5. Intent Preservation
        intent_score = self._calculate_intent_preservation(
            hypothesis, original_output, simulated_output
        )
        scores["intent"] = intent_score
        
        # 6. Complexity Penalty
        complexity_penalty = self._calculate_complexity_penalty(hypothesis.prompt)
        scores["complexity"] = complexity_penalty
        
        # Calculate weighted score with dynamic weights
        weighted_score = 0
        normalization_factor = 0
        
        for component, score in scores.items():
            weight = weights.get(component, 0)
            if component != "complexity":  # Complexity is a penalty
                weighted_score += score * abs(weight)
                normalization_factor += abs(weight)
            else:
                weighted_score += score * weight  # Negative weight for penalty
        
        # Normalize
        if normalization_factor > 0:
            final_score = max(0, min(1, weighted_score / normalization_factor))
        else:
            final_score = 0
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(scores, weights)
        
        # Identify uncertainty factors
        uncertainty_factors = self._identify_uncertainty_factors(scores, semantic_components)
        
        # Create enhanced scoring object
        enhanced_scoring = EnhancedScoring(
            semantic_similarity=semantic_score,
            structural_match=structural_score,
            constraint_satisfaction=constraint_score,
            style_match=style_score,
            intent_preservation=intent_score,
            complexity_penalty=complexity_penalty,
            semantic_components=semantic_components,
            structural_components=structural_components,
            confidence_score=final_score,
            confidence_interval=confidence_interval,
            uncertainty_factors=uncertainty_factors,
            weights=weights
        )
        
        # Identify missing and extra elements
        missing_elements, extra_elements = self._identify_differences(
            hypothesis, original_output
        )
        
        # Calculate confidence calibration
        confidence_calibration = self._calculate_confidence_calibration(
            hypothesis.confidence, final_score
        )
        
        # Create validation result with enhanced scoring
        result = ValidationResult(
            hypothesis=hypothesis.prompt,
            match_score=final_score,
            semantic_similarity=semantic_score,
            structural_match=structural_score,
            constraint_satisfaction=constraint_score,
            missing_elements=missing_elements,
            extra_elements=extra_elements,
            confidence_calibration=confidence_calibration
        )
        
        # Add enhanced scoring to result (extending the model)
        result.enhanced_scoring = enhanced_scoring
        
        return result
    
    def _detect_prompt_type(self, prompt: str) -> str:
        """Detect the type of prompt for dynamic weight selection"""
        prompt_lower = prompt.lower()
        
        # Technical indicators
        technical_keywords = ["explain", "algorithm", "function", "code", "implement", 
                            "technical", "system", "api", "database", "architecture"]
        technical_score = sum(1 for kw in technical_keywords if kw in prompt_lower)
        
        # Creative indicators
        creative_keywords = ["story", "creative", "imagine", "describe", "write", 
                           "narrative", "character", "plot", "scene", "poem"]
        creative_score = sum(1 for kw in creative_keywords if kw in prompt_lower)
        
        # Instructional indicators
        instructional_keywords = ["list", "steps", "how to", "guide", "tutorial", 
                                "instructions", "enumerate", "outline", "summarize"]
        instructional_score = sum(1 for kw in instructional_keywords if kw in prompt_lower)
        
        # Determine type based on scores
        scores = {
            "technical": technical_score,
            "creative": creative_score,
            "instructional": instructional_score
        }
        
        max_type = max(scores, key=scores.get)
        return max_type if scores[max_type] > 0 else "default"
    
    async def _calculate_enhanced_semantic_similarity(
        self, 
        text1: str, 
        text2: str
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate enhanced semantic similarity with component breakdown"""
        components = {}
        
        # 1. Embedding similarity (if available)
        embedding_sim = await self._calculate_embedding_similarity(text1, text2)
        components["embedding"] = embedding_sim
        
        # 2. N-gram overlap
        ngram_sim = self._calculate_ngram_similarity(text1, text2)
        components["ngram"] = ngram_sim
        
        # 3. Key concept overlap
        concept_sim = self._calculate_concept_similarity(text1, text2)
        components["concepts"] = concept_sim
        
        # 4. Semantic role similarity
        role_sim = self._calculate_semantic_role_similarity(text1, text2)
        components["semantic_roles"] = role_sim
        
        # Weighted combination
        weights = {
            "embedding": 0.4,
            "ngram": 0.2,
            "concepts": 0.25,
            "semantic_roles": 0.15
        }
        
        total_score = sum(components[k] * weights[k] for k in weights if k in components)
        
        return total_score, components
    
    def _calculate_enhanced_structural_match(
        self, 
        original: str, 
        simulated: str
    ) -> Tuple[float, Dict[str, float]]:
        """Enhanced structural matching with detailed components"""
        components = {}
        
        # Extract features for both texts
        orig_features = self._extract_detailed_structural_features(original)
        sim_features = self._extract_detailed_structural_features(simulated)
        
        # 1. Format similarity
        format_score = self._compare_formats(orig_features, sim_features)
        components["format"] = format_score
        
        # 2. Length similarity (with tolerance)
        length_score = self._calculate_length_similarity(orig_features, sim_features)
        components["length"] = length_score
        
        # 3. Organization similarity
        org_score = self._calculate_organization_similarity(orig_features, sim_features)
        components["organization"] = org_score
        
        # 4. Special elements
        special_score = self._calculate_special_elements_match(orig_features, sim_features)
        components["special_elements"] = special_score
        
        # Weighted combination
        total_score = (
            format_score * 0.3 +
            length_score * 0.25 +
            org_score * 0.25 +
            special_score * 0.2
        )
        
        return total_score, components
    
    async def _calculate_style_match(self, original: str, simulated: str) -> float:
        """Calculate style and tone matching score"""
        # Extract style features
        orig_style = self._extract_style_features(original)
        sim_style = self._extract_style_features(simulated)
        
        # Compare formality
        formality_diff = abs(orig_style["formality"] - sim_style["formality"])
        formality_score = 1 - (formality_diff / 10)  # Assuming 0-10 scale
        
        # Compare sentiment
        sentiment_diff = abs(orig_style["sentiment"] - sim_style["sentiment"])
        sentiment_score = 1 - sentiment_diff
        
        # Compare complexity
        complexity_diff = abs(orig_style["complexity"] - sim_style["complexity"])
        complexity_score = 1 - (complexity_diff / 10)
        
        # Weighted average
        style_score = (
            formality_score * 0.4 +
            sentiment_score * 0.3 +
            complexity_score * 0.3
        )
        
        return max(0, min(1, style_score))
    
    def _calculate_intent_preservation(
        self, 
        hypothesis: PromptHypothesis,
        original: str,
        simulated: str
    ) -> float:
        """Calculate how well the intent is preserved"""
        # Check if key elements from hypothesis appear in both outputs
        key_elements_score = 0
        for element in hypothesis.key_elements:
            element_lower = element.lower()
            if element_lower in original.lower() and element_lower in simulated.lower():
                key_elements_score += 1
        
        if hypothesis.key_elements:
            key_elements_score /= len(hypothesis.key_elements)
        else:
            key_elements_score = 0.5
        
        # Check if the main purpose is preserved
        purpose_keywords = self._extract_purpose_keywords(hypothesis.prompt)
        purpose_score = self._calculate_purpose_preservation(
            purpose_keywords, original, simulated
        )
        
        # Combined score
        return key_elements_score * 0.6 + purpose_score * 0.4
    
    def _calculate_complexity_penalty(self, prompt: str) -> float:
        """Calculate penalty for overly complex prompts"""
        # Factors that increase complexity
        word_count = len(prompt.split())
        sentence_count = len(re.split(r'[.!?]+', prompt))
        
        # Penalty factors
        length_penalty = min(word_count / 100, 1.0)  # Penalty for long prompts
        
        # Check for multiple instructions
        instruction_markers = ["and", "also", "then", "additionally", "furthermore"]
        multi_instruction_penalty = sum(1 for marker in instruction_markers if marker in prompt.lower()) / 10
        
        # Check for nested conditions
        conditional_words = ["if", "when", "unless", "except", "but"]
        conditional_penalty = sum(1 for word in conditional_words if word in prompt.lower()) / 10
        
        # Total penalty (higher means more complex, worse)
        total_penalty = min(1.0, (length_penalty * 0.4 + multi_instruction_penalty * 0.3 + conditional_penalty * 0.3))
        
        # Return as a score where 1 is best (simple) and 0 is worst (complex)
        return 1 - total_penalty
    
    def _calculate_confidence_interval(
        self, 
        scores: Dict[str, float], 
        weights: Dict[str, float]
    ) -> Tuple[float, float]:
        """Calculate 95% confidence interval for the score"""
        # Simulate score distribution based on component variance
        score_values = list(scores.values())
        
        if len(score_values) < 2:
            return (0.0, 1.0)
        
        # Calculate standard deviation of scores
        std_dev = statistics.stdev(score_values)
        mean_score = statistics.mean(score_values)
        
        # Approximate 95% CI (mean ± 1.96 * std_dev)
        margin = 1.96 * std_dev / np.sqrt(len(score_values))
        
        lower_bound = max(0, mean_score - margin)
        upper_bound = min(1, mean_score + margin)
        
        return (lower_bound, upper_bound)
    
    def _identify_uncertainty_factors(
        self, 
        scores: Dict[str, float], 
        semantic_components: Dict[str, float]
    ) -> List[str]:
        """Identify factors contributing to uncertainty"""
        factors = []
        
        # Low individual scores
        for component, score in scores.items():
            if score < 0.5:
                factors.append(f"Low {component} score ({score:.2f})")
        
        # High variance between scores
        score_values = list(scores.values())
        if len(score_values) > 1:
            variance = statistics.variance(score_values)
            if variance > 0.1:
                factors.append(f"High variance between scoring components ({variance:.3f})")
        
        # Missing semantic components
        if semantic_components.get("embedding", 0) < 0.1:
            factors.append("No embedding similarity available")
        
        # Extreme complexity
        if scores.get("complexity", 1) < 0.3:
            factors.append("High prompt complexity")
        
        return factors
    
    async def _generate_comparison(
        self, 
        hypothesis: PromptHypothesis,
        original_output: str
    ) -> PromptComparison:
        """Generate side-by-side comparison"""
        # Generate simulated output
        simulated_output = await self._simulate_output_from_prompt(hypothesis.prompt)
        
        # Identify differences and similarities
        differences = self._identify_key_differences(original_output, simulated_output)
        similarities = self._identify_key_similarities(original_output, simulated_output)
        
        return PromptComparison(
            hypothesis_prompt=hypothesis.prompt,
            simulated_output=simulated_output,
            original_output=original_output,
            differences=differences,
            similarities=similarities
        )
    
    def _identify_key_differences(self, text1: str, text2: str) -> List[str]:
        """Identify key differences between two texts"""
        differences = []
        
        # Length difference
        len_diff = abs(len(text1) - len(text2))
        if len_diff > 50:
            differences.append(f"Length differs by {len_diff} characters")
        
        # Format differences
        if bool(re.search(r'^\d+[.)]\s+', text1, re.MULTILINE)) != bool(re.search(r'^\d+[.)]\s+', text2, re.MULTILINE)):
            differences.append("Different list formatting")
        
        # Content differences
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        unique_to_1 = words1 - words2
        unique_to_2 = words2 - words1
        
        if len(unique_to_1) > 5:
            differences.append(f"Original has {len(unique_to_1)} unique words")
        if len(unique_to_2) > 5:
            differences.append(f"Simulated has {len(unique_to_2)} unique words")
        
        return differences
    
    def _identify_key_similarities(self, text1: str, text2: str) -> List[str]:
        """Identify key similarities between two texts"""
        similarities = []
        
        # Common structure
        if "list" in text1.lower() and "list" in text2.lower():
            similarities.append("Both contain list structures")
        
        # Common keywords
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        common_words = words1.intersection(words2)
        
        if len(common_words) > 10:
            similarities.append(f"{len(common_words)} words in common")
        
        # Similar length
        if abs(len(text1) - len(text2)) < 50:
            similarities.append("Similar length")
        
        return similarities
    
    # Additional helper methods...
    
    def _extract_style_features(self, text: str) -> Dict[str, float]:
        """Extract style features from text"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        # Formality (0-10 scale)
        formal_words = ["therefore", "however", "furthermore", "consequently", "whereas"]
        informal_words = ["gonna", "wanna", "yeah", "ok", "cool"]
        
        formality_score = sum(1 for w in formal_words if w in text.lower())
        formality_score -= sum(1 for w in informal_words if w in text.lower())
        formality_score = max(0, min(10, formality_score + 5))
        
        # Sentiment (simplified: -1 to 1)
        positive_words = ["good", "great", "excellent", "wonderful", "amazing", "benefit"]
        negative_words = ["bad", "poor", "terrible", "awful", "problem", "issue"]
        
        sentiment = sum(1 for w in positive_words if w in text.lower())
        sentiment -= sum(1 for w in negative_words if w in text.lower())
        sentiment = max(-1, min(1, sentiment / 10))
        
        # Complexity (0-10 scale based on average sentence length)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        complexity = min(10, avg_sentence_length / 3)
        
        return {
            "formality": formality_score,
            "sentiment": sentiment,
            "complexity": complexity
        }
    
    # Inherit remaining methods from the original validator...
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _simulate_output_from_prompt(self, prompt: str) -> str:
        """Simulate what output would be generated from the prompt"""
        if self.client is None:
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
                    prompt=prompt,
                    max_tokens=300,
                    temperature=0.3,
                    truncate='END'
                )
                if hasattr(response, 'generations') and len(response.generations) > 0:
                    return response.generations[0].text.strip()
                else:
                    return str(response).strip()
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
    
    # Additional helper methods
    
    async def _calculate_embedding_similarity(self, text1: str, text2: str) -> float:
        """Calculate embedding-based similarity"""
        if self.client is None:
            return self._simple_text_similarity(text1, text2)
            
        try:
            # Check if using ClientV2 or legacy Client
            if hasattr(cohere, 'ClientV2') and isinstance(self.client, cohere.ClientV2):
                # ClientV2 API
                response = self.client.embed(
                    texts=[text1[:500], text2[:500]],
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
                    texts=[text1[:500], text2[:500]]  # Limit length
                    # model parameter not needed for legacy embed
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
            self.logger.error(f"Failed to calculate embedding similarity: {str(e)}")
        
        # Fallback to simple similarity
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
    
    def _calculate_ngram_similarity(self, text1: str, text2: str, n=2) -> float:
        """Calculate n-gram similarity"""
        def get_ngrams(text, n):
            words = text.lower().split()
            return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))
        
        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_concept_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on key concepts"""
        # Extract key concepts (simplified version)
        concepts1 = self._extract_concepts(text1)
        concepts2 = self._extract_concepts(text2)
        
        if not concepts1 or not concepts2:
            return 0.0
        
        common = concepts1.intersection(concepts2)
        total = concepts1.union(concepts2)
        
        return len(common) / len(total) if total else 0.0
    
    def _extract_concepts(self, text: str) -> set:
        """Extract key concepts from text"""
        # Simple implementation: extract nouns and important phrases
        import re
        
        # Common concept patterns
        concepts = set()
        
        # Extract capitalized words (likely important)
        capitals = re.findall(r'\b[A-Z][a-z]+\b', text)
        concepts.update(c.lower() for c in capitals)
        
        # Extract words in quotes
        quoted = re.findall(r'"([^"]+)"', text)
        concepts.update(quoted)
        
        # Extract technical terms (words with special characters)
        technical = re.findall(r'\b\w+[-_]\w+\b', text)
        concepts.update(technical)
        
        return concepts
    
    def _calculate_semantic_role_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on semantic roles"""
        # Simplified version: check for similar sentence structures
        sentences1 = re.split(r'[.!?]+', text1)
        sentences2 = re.split(r'[.!?]+', text2)
        
        # Check for similar sentence patterns
        pattern_score = 0
        patterns = [
            r'^(What|When|Where|Who|How|Why)',  # Questions
            r'^(First|Second|Third|Finally)',   # Sequential
            r'(because|therefore|however)',      # Causal/contrast
        ]
        
        for pattern in patterns:
            count1 = sum(1 for s in sentences1 if re.search(pattern, s, re.I))
            count2 = sum(1 for s in sentences2 if re.search(pattern, s, re.I))
            
            if count1 > 0 and count2 > 0:
                pattern_score += min(count1, count2) / max(count1, count2)
        
        return pattern_score / len(patterns) if patterns else 0.0
    
    def _extract_detailed_structural_features(self, text: str) -> Dict[str, Any]:
        """Extract detailed structural features"""
        features = {}
        
        # Basic counts
        lines = text.split('\n')
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        features['line_count'] = len([l for l in lines if l.strip()])
        features['word_count'] = len(words)
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        features['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
        
        # List detection
        features['has_numbered_list'] = bool(re.search(r'^\d+[.)]\s+', text, re.MULTILINE))
        features['has_bullet_list'] = bool(re.search(r'^[-*+]\s+', text, re.MULTILINE))
        features['list_item_count'] = len(re.findall(r'^[\d\-*+][.)]*\s+', text, re.MULTILINE))
        
        # Special elements
        features['has_code'] = '```' in text or bool(re.search(r'^\s{4,}', text, re.MULTILINE))
        features['has_quotes'] = '"' in text or "'" in text
        features['has_questions'] = '?' in text
        features['has_headings'] = bool(re.search(r'^#+\s+', text, re.MULTILINE))
        
        # Structure type
        if features['has_numbered_list']:
            features['structure_type'] = 'numbered_list'
        elif features['has_bullet_list']:
            features['structure_type'] = 'bullet_list'
        elif features['paragraph_count'] > 3:
            features['structure_type'] = 'essay'
        else:
            features['structure_type'] = 'paragraph'
        
        return features
    
    def _compare_formats(self, features1: Dict, features2: Dict) -> float:
        """Compare format similarity"""
        if features1.get('structure_type') == features2.get('structure_type'):
            return 1.0
        
        # Partial credit for similar formats
        similar_formats = {
            ('numbered_list', 'bullet_list'): 0.7,
            ('bullet_list', 'numbered_list'): 0.7,
            ('essay', 'paragraph'): 0.5,
            ('paragraph', 'essay'): 0.5,
        }
        
        key = (features1.get('structure_type'), features2.get('structure_type'))
        return similar_formats.get(key, 0.0)
    
    def _calculate_length_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate length similarity with tolerance"""
        scores = []
        
        # Compare different length metrics
        metrics = ['word_count', 'line_count', 'sentence_count']
        
        for metric in metrics:
            val1 = features1.get(metric, 0)
            val2 = features2.get(metric, 0)
            
            if val1 == 0 and val2 == 0:
                scores.append(1.0)
            elif val1 == 0 or val2 == 0:
                scores.append(0.0)
            else:
                # Allow 20% tolerance
                ratio = min(val1, val2) / max(val1, val2)
                scores.append(ratio)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_organization_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate organizational similarity"""
        org_score = 0.0
        factors = 0
        
        # Paragraph structure
        if features1.get('paragraph_count', 0) == features2.get('paragraph_count', 0):
            org_score += 1.0
        factors += 1
        
        # List presence
        if features1.get('has_numbered_list') == features2.get('has_numbered_list'):
            org_score += 1.0
        factors += 1
        
        if features1.get('has_bullet_list') == features2.get('has_bullet_list'):
            org_score += 1.0
        factors += 1
        
        return org_score / factors if factors > 0 else 0.0
    
    def _calculate_special_elements_match(self, features1: Dict, features2: Dict) -> float:
        """Calculate match for special elements"""
        elements = ['has_code', 'has_quotes', 'has_questions', 'has_headings']
        matches = sum(1 for e in elements if features1.get(e, False) == features2.get(e, False))
        
        return matches / len(elements) if elements else 0.0
    
    def _calculate_enhanced_constraint_satisfaction(
        self, 
        hypothesis: PromptHypothesis,
        output: str
    ) -> float:
        """Enhanced constraint satisfaction checking"""
        satisfied = 0
        total = 0
        
        prompt_lower = hypothesis.prompt.lower()
        
        # Number constraints
        numbers = re.findall(r'\b(\d+)\b', hypothesis.prompt)
        for num in numbers:
            total += 1
            num_int = int(num)
            
            # Check various number-related constraints
            list_items = re.findall(r'^[\d\-*+][.)]*\s+', output, re.MULTILINE)
            word_count = len(output.split())
            line_count = len([l for l in output.split('\n') if l.strip()])
            
            # Flexible matching
            if len(list_items) == num_int:
                satisfied += 1
            elif abs(word_count - num_int * 10) < num_int * 5:  # Rough word count
                satisfied += 0.5
            elif abs(line_count - num_int) <= 2:
                satisfied += 0.5
        
        # Format constraints
        format_keywords = {
            'list': r'^[\d\-*+][.)]*\s+',
            'bullet': r'^[-*+]\s+',
            'number': r'^\d+[.)]\s+',
            'paragraph': r'\n\n',
            'question': r'\?',
            'step': r'(step|first|second|then|finally)',
        }
        
        for keyword, pattern in format_keywords.items():
            if keyword in prompt_lower:
                total += 1
                if re.search(pattern, output, re.MULTILINE | re.IGNORECASE):
                    satisfied += 1
        
        # Length constraints
        length_keywords = {
            'brief': (0, 100),
            'short': (50, 200),
            'detailed': (200, 1000),
            'comprehensive': (500, 2000),
        }
        
        for keyword, (min_words, max_words) in length_keywords.items():
            if keyword in prompt_lower:
                total += 1
                word_count = len(output.split())
                if min_words <= word_count <= max_words:
                    satisfied += 1
                elif min_words <= word_count <= max_words * 1.5:  # Some tolerance
                    satisfied += 0.5
        
        return satisfied / max(total, 1)
    
    def _extract_purpose_keywords(self, prompt: str) -> List[str]:
        """Extract purpose/intent keywords from prompt"""
        purpose_verbs = [
            'explain', 'describe', 'list', 'analyze', 'compare',
            'summarize', 'evaluate', 'define', 'illustrate', 'demonstrate'
        ]
        
        keywords = []
        prompt_lower = prompt.lower()
        
        for verb in purpose_verbs:
            if verb in prompt_lower:
                keywords.append(verb)
        
        return keywords
    
    def _calculate_purpose_preservation(
        self, 
        purpose_keywords: List[str],
        original: str,
        simulated: str
    ) -> float:
        """Calculate how well the purpose is preserved"""
        if not purpose_keywords:
            return 0.5  # Neutral if no clear purpose
        
        score = 0.0
        
        for keyword in purpose_keywords:
            # Check if outputs reflect the purpose
            if keyword == 'list':
                if re.search(r'^[\d\-*+][.)]*\s+', original, re.MULTILINE) and \
                   re.search(r'^[\d\-*+][.)]*\s+', simulated, re.MULTILINE):
                    score += 1
            elif keyword == 'explain':
                # Check for explanatory language
                explain_words = ['because', 'therefore', 'thus', 'means', 'is']
                orig_has_explain = any(w in original.lower() for w in explain_words)
                sim_has_explain = any(w in simulated.lower() for w in explain_words)
                if orig_has_explain and sim_has_explain:
                    score += 1
            # Add more purpose checks as needed
            else:
                # Generic check: keyword appears in both
                if keyword in original.lower() and keyword in simulated.lower():
                    score += 0.5
        
        return score / len(purpose_keywords)
    
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
        output_features = self._extract_detailed_structural_features(output)
        
        if output_features.get('has_code') and 'code' not in hypothesis.prompt.lower():
            extra_elements.append("Unexpected code blocks")
        
        if output_features.get('has_questions') and 'question' not in hypothesis.prompt.lower():
            extra_elements.append("Unexpected questions")
        
        # Check for format mismatches
        if 'list' in hypothesis.prompt.lower() and not output_features.get('has_numbered_list') and not output_features.get('has_bullet_list'):
            missing_elements.append("Expected list format")
        
        return missing_elements, extra_elements
    
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