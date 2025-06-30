import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import logging
from tools import PatternAnalyzer, GeminiHypothesisGenerator, CohereValidator
from tools.enhanced_validator import EnhancedValidator
from models import (
    AnalysisReport, HypothesisSet, ValidationResult,
    DetectionResult, PromptHypothesis, ConfidenceLevel
)
from models.responses import PromptComparison, EnhancedScoring

class PromptDetectiveAgent:
    """Main agent that orchestrates the prompt detection process"""
    
    SYSTEM_PROMPT = """You are the Prompt Detective, an expert AI agent specialized in reverse engineering prompts from their outputs.

Your mission is to analyze AI-generated text and deduce the original prompt that created it through systematic investigation.

Core Principles:
1. Think like a detective - gather evidence before making conclusions
2. Consider multiple hypotheses and test each one
3. Be confident when evidence is strong, uncertain when it's weak
4. Learn from failed attempts and refine your approach
5. Explain your reasoning clearly

Process:
- First, analyze the output for patterns and constraints
- Generate multiple prompt hypotheses based on evidence
- Validate each hypothesis through testing
- Refine based on validation results
- Present your findings with appropriate confidence

Remember: Your goal is accuracy, not speed. It's better to admit uncertainty than to be confidently wrong."""
    
    def __init__(self):
        self.logger = logging.getLogger("agent.prompt_detective")
        self.pattern_analyzer = PatternAnalyzer()
        self.hypothesis_generator = GeminiHypothesisGenerator()
        self.validator = CohereValidator()
        self.enhanced_validator = EnhancedValidator()
        
    async def detect_prompt(
        self,
        output_text: str,
        max_attempts: int = 3,
        context: Optional[str] = None
    ) -> DetectionResult:
        """Main detection method with multi-pass refinement"""
        start_time = time.time()
        execution_trace = []
        all_hypotheses = []
        validation_results = []
        thinking_process = []
        
        # Apply system prompt context
        self.logger.info(f"Starting prompt detection with system prompt guidance")
        
        try:
            # Pass 1: Initial Analysis
            self.logger.info("Pass 1: Initial analysis")
            thinking_process.append({
                "stage": "Initial Assessment",
                "thoughts": [
                    f"Received output with {len(output_text)} characters",
                    "Need to identify patterns, format, and constraints",
                    "Will analyze linguistic features and structure first"
                ]
            })
            
            trace_entry = {
                "pass": 1, 
                "action": "initial_analysis", 
                "timestamp": datetime.utcnow().isoformat(),
                "thinking": "Analyzing the output structure and patterns to understand what kind of prompt could have generated this."
            }
            
            self.logger.info("About to call pattern analyzer...")
            analysis = await self.pattern_analyzer.execute(output_text)
            self.logger.info("Pattern analyzer completed")
            trace_entry["analysis_confidence"] = analysis.overall_confidence.value
            
            # Agent's reasoning about the analysis
            pattern_thoughts = []
            if analysis.structural_features.get('format_type'):
                pattern_thoughts.append(f"Detected {analysis.structural_features['format_type']} format - this suggests a structured prompt")
            if analysis.constraints_detected:
                pattern_thoughts.append(f"Found constraints: {', '.join(analysis.constraints_detected)} - prompt likely specified these")
            if analysis.linguistic_features.get('tone'):
                pattern_thoughts.append(f"Tone is {analysis.linguistic_features['tone']} - prompt may have requested this style")
            
            thinking_process.append({
                "stage": "Pattern Analysis Complete",
                "thoughts": pattern_thoughts
            })
            trace_entry["pattern_insights"] = pattern_thoughts
            execution_trace.append(trace_entry)
            
            # Generate initial hypotheses
            thinking_process.append({
                "stage": "Hypothesis Generation",
                "thoughts": [
                    "Based on patterns, I need to generate likely prompt candidates",
                    f"Found {len(analysis.patterns)} patterns and {len(analysis.constraints_detected)} constraints",
                    "Will use Gemini to generate creative variations"
                ]
            })
            
            hypothesis_input = {
                'analysis_report': analysis,
                'output_text': output_text,
                'iteration': 1
            }
            
            hypothesis_set = await self.hypothesis_generator.execute(hypothesis_input)
            all_hypotheses.extend(hypothesis_set.hypotheses)
            
            # Agent's reasoning about hypotheses
            hyp_thoughts = []
            for h in hypothesis_set.hypotheses[:2]:
                hyp_thoughts.append(f"'{h.prompt}' - confidence {h.confidence:.2f} because {h.reasoning}")
            
            thinking_process.append({
                "stage": "Hypotheses Generated",
                "thoughts": hyp_thoughts
            })
            
            # Validate top hypotheses
            thinking_process.append({
                "stage": "Validation Strategy",
                "thoughts": [
                    "Will validate top 2 hypotheses using semantic similarity",
                    "Cohere will help determine if these prompts would generate similar output",
                    "Looking for match score > 0.85 for high confidence"
                ]
            })
            
            top_hypotheses = hypothesis_set.hypotheses[:2]  # Validate top 2
            validation_input = {
                'hypothesis': top_hypotheses,
                'original_output': output_text
            }
            
            initial_validations = await self.validator.execute(validation_input)
            validation_results.extend(initial_validations)
            
            # Check if we have a good match
            best_validation = max(initial_validations, key=lambda v: v.match_score)
            
            val_thoughts = []
            for v in initial_validations:
                feedback = "Strong match" if v.match_score > 0.8 else "Moderate match" if v.match_score > 0.6 else "Weak match"
                val_thoughts.append(f"'{v.hypothesis}' scored {v.match_score:.2f} - {feedback}")
            
            thinking_process.append({
                "stage": "Validation Results",
                "thoughts": val_thoughts
            })
            
            if best_validation.match_score >= 0.85:
                self.logger.info(f"High confidence match found in Pass 1: {best_validation.match_score}")
                thinking_process.append({
                    "stage": "Decision",
                    "thoughts": [
                        f"Match score {best_validation.match_score:.2f} exceeds threshold",
                        "High confidence in this detection - no refinement needed",
                        f"Final answer: '{best_validation.hypothesis}'"
                    ]
                })
                return self._create_result(
                    all_hypotheses, validation_results, 1, execution_trace, start_time, thinking_process
                )
            
            # Pass 2: Refinement
            if max_attempts >= 2:
                self.logger.info("Pass 2: Refinement based on validation feedback")
                
                thinking_process.append({
                    "stage": "Refinement Decision",
                    "thoughts": [
                        f"Best score {best_validation.match_score:.2f} is below 0.85 threshold",
                        "Need to refine hypotheses based on validation feedback",
                        "Will analyze what elements might be missing"
                    ]
                })
                
                trace_entry = {
                    "pass": 2, 
                    "action": "refinement", 
                    "timestamp": datetime.utcnow().isoformat(),
                    "thinking": "Scores are not high enough. Need to understand what's missing and refine."
                }
                
                # Identify what's missing
                missing_elements = []
                for val in initial_validations:
                    missing_elements.extend(val.missing_elements)
                
                thinking_process.append({
                    "stage": "Missing Elements Analysis",
                    "thoughts": [
                        f"Validation identified missing elements: {', '.join(missing_elements) if missing_elements else 'none specific'}",
                        "Previous hypotheses may have been too general or missed key constraints",
                        "Will generate more specific variations"
                    ]
                })
                
                trace_entry["missing_elements"] = missing_elements
                execution_trace.append(trace_entry)
                
                # Refine with feedback
                refinement_input = {
                    'analysis_report': analysis,
                    'output_text': output_text,
                    'previous_attempts': [h.prompt for h in top_hypotheses],
                    'iteration': 2
                }
                
                refined_set = await self.hypothesis_generator.execute(refinement_input)
                all_hypotheses.extend(refined_set.hypotheses)
                
                ref_thoughts = []
                for h in refined_set.hypotheses[:2]:
                    ref_thoughts.append(f"Refined: '{h.prompt}' - addresses {h.reasoning}")
                
                thinking_process.append({
                    "stage": "Refined Hypotheses",
                    "thoughts": ref_thoughts
                })
                
                # Validate refined hypotheses
                refined_validations = await self.validator.execute({
                    'hypothesis': refined_set.hypotheses[:2],
                    'original_output': output_text
                })
                validation_results.extend(refined_validations)
                
                best_validation = max(validation_results, key=lambda v: v.match_score)
                
                thinking_process.append({
                    "stage": "Refinement Results",
                    "thoughts": [
                        f"Best refined score: {best_validation.match_score:.2f}",
                        f"Improvement from Pass 1: {best_validation.match_score - initial_validations[0].match_score:.2f}",
                        "Evaluating if further refinement is needed..."
                    ]
                })
                
                if best_validation.match_score >= 0.8 or max_attempts == 2:
                    thinking_process.append({
                        "stage": "Pass 2 Decision",
                        "thoughts": [
                            f"Score {best_validation.match_score:.2f} is {'acceptable' if best_validation.match_score >= 0.8 else 'the best we can achieve'}",
                            f"Stopping at Pass 2 with: '{best_validation.hypothesis}'"
                        ]
                    })
                    return self._create_result(
                        all_hypotheses, validation_results, 2, execution_trace, start_time, thinking_process
                    )
            
            # Pass 3: Deep analysis
            if max_attempts >= 3:
                self.logger.info("Pass 3: Deep analysis with context")
                
                thinking_process.append({
                    "stage": "Pass 3 Deep Analysis",
                    "thoughts": [
                        f"After 2 passes, best score is {best_validation.match_score:.2f}",
                        "Will perform deeper analysis with context",
                        "Examining subtle patterns and edge cases"
                    ]
                })
                
                trace_entry = {
                    "pass": 3, 
                    "action": "deep_analysis", 
                    "timestamp": datetime.utcnow().isoformat(),
                    "thinking": "Deep analysis with accumulated context and pattern recognition."
                }
                
                # Synthesize learnings from all attempts
                all_scores = [v.match_score for v in validation_results]
                trace_entry["previous_scores"] = all_scores
                execution_trace.append(trace_entry)
                
                thinking_process.append({
                    "stage": "Learning Synthesis",
                    "thoughts": [
                        f"Previous scores: {[f'{s:.2f}' for s in all_scores]}",
                        "Analyzing what worked and what didn't",
                        "Combining best elements from all attempts"
                    ]
                })
                
                # Final attempt with all context
                final_input = {
                    'analysis_report': analysis,
                    'output_text': output_text,
                    'previous_attempts': [h.prompt for h in all_hypotheses],
                    'iteration': 3
                }
                
                final_set = await self.hypothesis_generator.execute(final_input)
                all_hypotheses.extend(final_set.hypotheses)
                
                thinking_process.append({
                    "stage": "Final Hypothesis",
                    "thoughts": [
                        f"Final synthesis: '{final_set.hypotheses[0].prompt}'",
                        f"This combines learnings from {len(all_hypotheses)-1} previous attempts",
                        "Validating final hypothesis..."
                    ]
                })
                
                # Final validation
                final_validations = await self.validator.execute({
                    'hypothesis': final_set.hypotheses[:1],  # Just the best one
                    'original_output': output_text
                })
                validation_results.extend(final_validations)
                
                final_score = final_validations[0].match_score
                thinking_process.append({
                    "stage": "Pass 3 Result",
                    "thoughts": [
                        f"Pass 3 score: {final_score:.2f}",
                        f"Improvement from Pass 2: {final_score - best_validation.match_score:.2f}",
                        "Evaluating if further attempts needed..."
                    ]
                })
                
                best_validation = max(validation_results, key=lambda v: v.match_score)
                
                if best_validation.match_score >= 0.85 or max_attempts == 3:
                    return await self._create_result(
                        all_hypotheses, validation_results, 3, execution_trace, start_time, thinking_process, output_text
                    )
            
            # Pass 4: Advanced pattern matching
            if max_attempts >= 4:
                self.logger.info("Pass 4: Advanced pattern matching and edge case analysis")
                
                thinking_process.append({
                    "stage": "Pass 4 Advanced Analysis",
                    "thoughts": [
                        f"After 3 passes, best score is {best_validation.match_score:.2f}",
                        "Applying advanced pattern matching techniques",
                        "Looking for subtle linguistic cues and hidden constraints"
                    ]
                })
                
                trace_entry = {
                    "pass": 4, 
                    "action": "advanced_pattern_matching", 
                    "timestamp": datetime.utcnow().isoformat(),
                    "thinking": "Advanced pattern matching for edge cases and subtle cues."
                }
                execution_trace.append(trace_entry)
                
                # Pass 4 with advanced context
                advanced_input = {
                    'analysis_report': analysis,
                    'output_text': output_text,
                    'previous_attempts': [h.prompt for h in all_hypotheses],
                    'iteration': 4
                }
                
                advanced_set = await self.hypothesis_generator.execute(advanced_input)
                all_hypotheses.extend(advanced_set.hypotheses)
                
                # Validate Pass 4 hypotheses
                advanced_validations = await self.validator.execute({
                    'hypothesis': advanced_set.hypotheses[:2],
                    'original_output': output_text
                })
                validation_results.extend(advanced_validations)
                
                best_validation = max(validation_results, key=lambda v: v.match_score)
                
                thinking_process.append({
                    "stage": "Pass 4 Results",
                    "thoughts": [
                        f"Pass 4 best score: {best_validation.match_score:.2f}",
                        "Checking if we've reached satisfactory confidence"
                    ]
                })
                
                if best_validation.match_score >= 0.9 or max_attempts == 4:
                    return await self._create_result(
                        all_hypotheses, validation_results, 4, execution_trace, start_time, thinking_process, output_text
                    )
            
            # Pass 5: Final synthesis with all learnings
            if max_attempts >= 5:
                self.logger.info("Pass 5: Final synthesis with complete context")
                
                thinking_process.append({
                    "stage": "Pass 5 Final Synthesis",
                    "thoughts": [
                        f"After 4 passes, best score is {best_validation.match_score:.2f}",
                        "Final attempt synthesizing all insights",
                        "This is the ultimate refinement opportunity"
                    ]
                })
                
                trace_entry = {
                    "pass": 5, 
                    "action": "final_synthesis", 
                    "timestamp": datetime.utcnow().isoformat(),
                    "thinking": "Final synthesis incorporating all learnings from previous attempts."
                }
                execution_trace.append(trace_entry)
                
                # Final synthesis with complete context
                final_input = {
                    'analysis_report': analysis,
                    'output_text': output_text,
                    'previous_attempts': [h.prompt for h in all_hypotheses],
                    'iteration': 5
                }
                
                final_set = await self.hypothesis_generator.execute(final_input)
                all_hypotheses.extend(final_set.hypotheses)
                
                # Final validation
                final_validations = await self.validator.execute({
                    'hypothesis': final_set.hypotheses[:1],
                    'original_output': output_text
                })
                validation_results.extend(final_validations)
                
                final_score = final_validations[0].match_score
                thinking_process.append({
                    "stage": "Final Result",
                    "thoughts": [
                        f"Final score after 5 passes: {final_score:.2f}",
                        f"Total improvement: {final_score - initial_validations[0].match_score:.2f}",
                        f"Confidence level: {self._determine_confidence(final_score).value}"
                    ]
                })
            
            # Return the best result we have
            return await self._create_result(
                all_hypotheses, validation_results, max_attempts, execution_trace, start_time, thinking_process, output_text
            )
            
        except Exception as e:
            self.logger.error(f"Detection failed: {str(e)}")
            raise
    
    async def _create_result(
        self,
        all_hypotheses: List[PromptHypothesis],
        validation_results: List[ValidationResult],
        attempts_used: int,
        execution_trace: List[Dict[str, Any]],
        start_time: float,
        thinking_process: List[Dict[str, Any]],
        output_text: str
    ) -> DetectionResult:
        """Create the final detection result with enhanced scoring and comparison"""
        
        # Find best hypothesis based on validation
        best_validation = max(validation_results, key=lambda v: v.match_score)
        best_hypothesis = next(
            h for h in all_hypotheses 
            if h.prompt == best_validation.hypothesis
        )
        
        # Run enhanced validation for the best hypothesis to get detailed scoring
        enhanced_result = await self.enhanced_validator.execute({
            'hypothesis': best_hypothesis,
            'original_output': output_text
        })
        
        # Extract comparison and enhanced scoring
        prompt_comparison = enhanced_result.get("comparison")
        enhanced_scoring = enhanced_result.get("enhanced_scoring")
        
        # Sort all hypotheses by confidence
        sorted_hypotheses = sorted(all_hypotheses, key=lambda h: h.confidence, reverse=True)
        
        # Determine overall confidence using enhanced scoring if available
        if enhanced_scoring:
            confidence = self._determine_confidence(enhanced_scoring.confidence_score)
        else:
            confidence = self._determine_confidence(best_validation.match_score)
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return DetectionResult(
            best_hypothesis=best_hypothesis,
            all_hypotheses=sorted_hypotheses,
            validation_results=validation_results,
            confidence=confidence,
            prompt_comparison=prompt_comparison,
            enhanced_scoring=enhanced_scoring,
            attempts_used=attempts_used,
            execution_trace=execution_trace,
            thinking_process=thinking_process,
            processing_time_ms=processing_time_ms
        )
    
    def _determine_confidence(self, match_score: float) -> ConfidenceLevel:
        """Determine confidence level from match score"""
        if match_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif match_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif match_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif match_score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW