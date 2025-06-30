import os
from typing import List, Dict, Any, Optional
from .base import Tool
from models.analysis import AnalysisReport
from models.hypothesis import PromptHypothesis, HypothesisSet
import json
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from google.cloud import aiplatform
    from google.oauth2 import service_account
    import vertexai
    # Always use preview for consistency
    from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False

class GeminiHypothesisGenerator(Tool):
    """Generates prompt hypotheses using Google's Gemini model"""
    
    def __init__(self):
        super().__init__(
            name="gemini_hypothesis_generator",
            description="Uses Gemini to generate likely prompt hypotheses"
        )
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize Vertex AI client"""
        if not VERTEX_AI_AVAILABLE:
            self.logger.warning("Vertex AI not available, using mock mode")
            self.model = None
            return
            
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
        
        if not project_id:
            self.logger.warning("GOOGLE_CLOUD_PROJECT not set, using mock mode")
            self.model = None
            return
        
        try:
            # Check for credentials
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if creds_path and os.path.exists(creds_path):
                self.logger.info(f"Using credentials from: {creds_path}")
                credentials = service_account.Credentials.from_service_account_file(creds_path)
                vertexai.init(project=project_id, location=location, credentials=credentials)
            else:
                # Try default credentials
                self.logger.info("Using default application credentials")
                vertexai.init(project=project_id, location=location)
            
            # Initialize the model
            self.logger.info("About to create GenerativeModel...")
            self.model = GenerativeModel("gemini-1.5-flash")
            self.logger.info("GenerativeModel created successfully")
            # Use default generation config for now
            # self.generation_config = GenerationConfig(
            #     temperature=0.7,
            #     top_p=0.9,
            #     max_output_tokens=1024,
            # )
            self.logger.info("Gemini model initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Vertex AI: {e}, using mock mode")
            self.model = None
    
    async def execute(self, input_data: Dict[str, Any]) -> HypothesisSet:
        """Generate prompt hypotheses based on analysis"""
        analysis_report = input_data.get('analysis_report')
        output_text = input_data.get('output_text')
        previous_attempts = input_data.get('previous_attempts', [])
        iteration = input_data.get('iteration', 1)
        
        # Generate hypotheses
        hypotheses = await self._generate_hypotheses(
            analysis_report, 
            output_text, 
            previous_attempts
        )
        
        # Rank hypotheses
        ranked_hypotheses = self._rank_hypotheses(hypotheses)
        
        return HypothesisSet(
            hypotheses=ranked_hypotheses,
            analysis_context={
                'patterns_found': len(analysis_report.patterns),
                'confidence_level': analysis_report.overall_confidence,
                'iteration': iteration
            },
            iteration=iteration
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _generate_hypotheses(
        self, 
        analysis: AnalysisReport, 
        output_text: str,
        previous_attempts: List[str]
    ) -> List[PromptHypothesis]:
        """Generate hypotheses using Gemini"""
        
        self.logger.info("Starting hypothesis generation...")
        prompt = self._build_prompt(analysis, output_text, previous_attempts)
        self.logger.info(f"Built prompt, length: {len(prompt)}")
        
        try:
            if self.model is None:
                # Use fallback if model not initialized
                self.logger.info("Model is None, using fallback...")
                return self._fallback_hypothesis_generation(analysis, output_text)
                
            # Try simplified call
            try:
                self.logger.info("Calling Gemini model...")
                response = self.model.generate_content(prompt)
                self.logger.info("Gemini response received")
                
                # Parse the response
                hypotheses = self._parse_response(response.text)
                self.logger.info(f"Parsed {len(hypotheses)} hypotheses from Gemini")
                return hypotheses
            except Exception as api_error:
                self.logger.error(f"Gemini API call failed: {api_error}")
                self.logger.error(f"Error type: {type(api_error).__name__}")
                # Fall back to rule-based generation
                self.logger.info("Falling back to rule-based generation")
                return self._fallback_hypothesis_generation(analysis, output_text)
            
        except Exception as e:
            self.logger.error(f"Gemini hypothesis generation error: {str(e)}")
            # Fallback to rule-based generation
            return self._fallback_hypothesis_generation(analysis, output_text)
    
    def _build_prompt(
        self, 
        analysis: AnalysisReport, 
        output_text: str,
        previous_attempts: List[str]
    ) -> str:
        """Build the prompt for Gemini"""
        
        # Extract key features
        features = []
        if analysis.structural_features.get('has_numbered_list'):
            features.append("numbered list")
        if analysis.structural_features.get('has_code'):
            features.append("code content")
        
        constraints_str = ", ".join(analysis.constraints_detected) if analysis.constraints_detected else "none detected"
        
        prompt = f"""You are an expert at reverse engineering prompts from AI-generated outputs.

IMPORTANT PRINCIPLES:
1. Apply Occam's Razor - simpler prompts are MORE LIKELY than complex ones
2. LLMs naturally add formatting (bullets, structure) without being asked
3. Most user prompts are under 15 words
4. Avoid assuming every output feature was explicitly requested
5. Start with the simplest prompt that could produce this output

Common simple patterns to consider FIRST:
- "Explain [topic]"  
- "What is [topic]?"
- "How does [topic] work?"
- "Write about [topic]"
- "Describe [topic]"
- "List [topic]"

Given the following AI-generated output:
---
{output_text[:500]}{'...' if len(output_text) > 500 else ''}
---

Analysis findings:
- Structural features: {features}
- Linguistic tone: {analysis.linguistic_features.get('formality_score', 0)}
- Constraints detected: {constraints_str}
- Word count: {analysis.linguistic_features.get('word_count', 0)}

{f"Previous attempts that were incorrect: {previous_attempts}" if previous_attempts else ""}

Generate 3 different prompt hypotheses that could have produced this output. 
IMPORTANT: Start with the SIMPLEST possible prompt, then add complexity only if truly necessary.

For each hypothesis, provide:
1. The exact prompt text (keep it SHORT and SIMPLE)
2. Confidence score (0-1)
3. Reasoning for why this prompt is likely
4. Key elements in the prompt

Format your response as a JSON array with objects containing: prompt, confidence, reasoning, key_elements

Remember: LLMs are helpful by default and will naturally structure responses well."""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> List[PromptHypothesis]:
        """Parse Gemini's response into hypothesis objects"""
        hypotheses = []
        
        try:
            # Extract JSON from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)
                
                for i, item in enumerate(parsed[:3]):  # Max 3 hypotheses
                    hypothesis = PromptHypothesis(
                        prompt=item.get('prompt', ''),
                        confidence=float(item.get('confidence', 0.5)),
                        reasoning=item.get('reasoning', ''),
                        key_elements=item.get('key_elements', []),
                        rank=i + 1
                    )
                    hypotheses.append(hypothesis)
            
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse JSON from Gemini response")
            # Try to extract hypotheses using regex as fallback
            hypotheses = self._extract_hypotheses_fallback(response_text)
        
        return hypotheses
    
    def _extract_hypotheses_fallback(self, response_text: str) -> List[PromptHypothesis]:
        """Fallback method to extract hypotheses from text"""
        hypotheses = []
        
        # Simple pattern matching for prompts in quotes
        import re
        prompt_pattern = r'"([^"]+)".*?confidence.*?(\d+\.?\d*)'
        matches = re.findall(prompt_pattern, response_text, re.IGNORECASE | re.DOTALL)
        
        for i, (prompt, confidence) in enumerate(matches[:3]):
            hypotheses.append(PromptHypothesis(
                prompt=prompt,
                confidence=float(confidence) if confidence else 0.5,
                reasoning="Extracted from response",
                key_elements=prompt.split()[:5],  # First 5 words as key elements
                rank=i + 1
            ))
        
        return hypotheses
    
    def _fallback_hypothesis_generation(
        self, 
        analysis: AnalysisReport, 
        output_text: str
    ) -> List[PromptHypothesis]:
        """Generate hypotheses using rules when API fails"""
        hypotheses = []
        
        # Rule-based generation based on patterns
        if analysis.structural_features.get('has_numbered_list'):
            count = analysis.structural_features.get('list_item_count', 5)
            hypotheses.append(PromptHypothesis(
                prompt=f"List {count} items about the topic",
                confidence=0.7,
                reasoning="Output contains a numbered list with specific count",
                key_elements=[f"List {count}", "items"],
                rank=1
            ))
        
        # Add more rule-based hypotheses based on detected patterns
        if analysis.constraints_detected:
            constraint_prompt = "Create output with: " + ", ".join(analysis.constraints_detected[:3])
            hypotheses.append(PromptHypothesis(
                prompt=constraint_prompt,
                confidence=0.6,
                reasoning="Based on detected constraints",
                key_elements=analysis.constraints_detected[:3],
                rank=2
            ))
        
        # Generic fallback
        if not hypotheses:
            hypotheses.append(PromptHypothesis(
                prompt="Generate content about the topic",
                confidence=0.3,
                reasoning="Generic fallback hypothesis",
                key_elements=["Generate", "content"],
                rank=3
            ))
        
        return hypotheses
    
    def _rank_hypotheses(self, hypotheses: List[PromptHypothesis]) -> List[PromptHypothesis]:
        """Rank hypotheses by confidence and adjust ranks"""
        # Sort by confidence
        sorted_hypotheses = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)
        
        # Update ranks
        for i, hypothesis in enumerate(sorted_hypotheses):
            hypothesis.rank = i + 1
        
        return sorted_hypotheses