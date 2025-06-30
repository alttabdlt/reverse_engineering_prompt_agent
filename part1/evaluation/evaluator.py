"""
Automated evaluation framework for the Prompt Detective system
"""
import asyncio
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime
import difflib
import re
from dataclasses import dataclass, asdict
import httpx
from .test_cases import TestCase, get_test_cases
from ..models import AnalyzeRequest

@dataclass
class EvaluationScore:
    """Detailed scoring for a test case"""
    test_id: str
    test_name: str
    exact_match: bool
    semantic_match_score: float  # 0-1
    key_elements_found: List[str]
    key_elements_missing: List[str]
    total_score: float  # 0-100
    confidence_calibration: float  # How well did it know when it was right/wrong
    passed: bool
    execution_time_ms: int
    detected_prompt: str
    original_prompt: str
    
@dataclass 
class EvaluationReport:
    """Overall evaluation report"""
    timestamp: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    average_score: float
    scores_by_category: Dict[str, float]
    scores_by_difficulty: Dict[str, float]
    individual_results: List[EvaluationScore]
    summary: str

class PromptEvaluator:
    """Evaluates the Prompt Detective system"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def evaluate_all(self, test_cases: List[TestCase] = None) -> EvaluationReport:
        """Run evaluation on all test cases"""
        if test_cases is None:
            test_cases = get_test_cases()
        
        print(f"Starting evaluation of {len(test_cases)} test cases...")
        
        results = []
        for test_case in test_cases:
            print(f"\nEvaluating: {test_case.name} ({test_case.category})")
            score = await self.evaluate_single(test_case)
            results.append(score)
            
            # Print immediate feedback
            status = "✓ PASSED" if score.passed else "✗ FAILED"
            print(f"  {status} - Score: {score.total_score:.1f}/100")
            print(f"  Detected: {score.detected_prompt[:50]}...")
        
        # Generate report
        report = self._generate_report(results)
        
        # Save report
        await self._save_report(report)
        
        return report
    
    async def evaluate_single(self, test_case: TestCase) -> EvaluationScore:
        """Evaluate a single test case"""
        start_time = asyncio.get_event_loop().time()
        
        # Call the API
        try:
            response = await self.client.post(
                f"{self.api_base_url}/analyze",
                json={
                    "output_text": test_case.generated_output,
                    "context": test_case.category,
                    "max_attempts": 3
                }
            )
            response.raise_for_status()
            result = response.json()
            
        except Exception as e:
            print(f"  API Error: {str(e)}")
            return EvaluationScore(
                test_id=test_case.id,
                test_name=test_case.name,
                exact_match=False,
                semantic_match_score=0.0,
                key_elements_found=[],
                key_elements_missing=test_case.expected_elements,
                total_score=0.0,
                confidence_calibration=0.0,
                passed=False,
                execution_time_ms=int((asyncio.get_event_loop().time() - start_time) * 1000),
                detected_prompt="[API Error]",
                original_prompt=test_case.original_prompt
            )
        
        # Extract detected prompt
        detected_prompt = result['result']['best_hypothesis']['prompt']
        confidence = result['result']['best_hypothesis']['confidence']
        
        # Calculate scores
        exact_match = self._check_exact_match(detected_prompt, test_case)
        semantic_score = self._calculate_semantic_score(detected_prompt, test_case)
        elements_found, elements_missing = self._check_key_elements(detected_prompt, test_case)
        
        # Calculate total score
        total_score = self._calculate_total_score(
            exact_match, semantic_score, elements_found, elements_missing, test_case
        )
        
        # Check confidence calibration
        confidence_calibration = self._check_confidence_calibration(
            confidence, total_score / 100
        )
        
        # Determine pass/fail
        passed = total_score >= 70  # 70% threshold
        
        execution_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        return EvaluationScore(
            test_id=test_case.id,
            test_name=test_case.name,
            exact_match=exact_match,
            semantic_match_score=semantic_score,
            key_elements_found=elements_found,
            key_elements_missing=elements_missing,
            total_score=total_score,
            confidence_calibration=confidence_calibration,
            passed=passed,
            execution_time_ms=execution_time,
            detected_prompt=detected_prompt,
            original_prompt=test_case.original_prompt
        )
    
    def _check_exact_match(self, detected: str, test_case: TestCase) -> bool:
        """Check if detected prompt exactly matches original or variations"""
        detected_lower = detected.lower().strip()
        
        # Check exact match
        if detected_lower == test_case.original_prompt.lower().strip():
            return True
        
        # Check acceptable variations
        for variation in test_case.acceptable_variations:
            if detected_lower == variation.lower().strip():
                return True
        
        return False
    
    def _calculate_semantic_score(self, detected: str, test_case: TestCase) -> float:
        """Calculate semantic similarity score"""
        # Use difflib for simple similarity
        original_words = set(test_case.original_prompt.lower().split())
        detected_words = set(detected.lower().split())
        
        # Also check variations
        all_valid_words = original_words.copy()
        for variation in test_case.acceptable_variations:
            all_valid_words.update(variation.lower().split())
        
        # Calculate Jaccard similarity
        intersection = detected_words.intersection(all_valid_words)
        union = detected_words.union(original_words)
        
        if not union:
            return 0.0
        
        jaccard = len(intersection) / len(union)
        
        # Also use sequence matcher for order similarity
        seq_score = difflib.SequenceMatcher(
            None, 
            test_case.original_prompt.lower(), 
            detected.lower()
        ).ratio()
        
        # Combine scores
        return (jaccard + seq_score) / 2
    
    def _check_key_elements(
        self, 
        detected: str, 
        test_case: TestCase
    ) -> Tuple[List[str], List[str]]:
        """Check which key elements were found"""
        detected_lower = detected.lower()
        found = []
        missing = []
        
        for element in test_case.expected_elements:
            element_lower = element.lower()
            
            # Check for exact word match or substring
            if (f" {element_lower} " in f" {detected_lower} " or
                element_lower in detected_lower or
                self._check_element_variation(element_lower, detected_lower)):
                found.append(element)
            else:
                missing.append(element)
        
        return found, missing
    
    def _check_element_variation(self, element: str, text: str) -> bool:
        """Check for variations of an element"""
        # Handle numbers
        if element.isdigit():
            # Check for written form (e.g., "5" -> "five")
            number_words = {
                "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
                "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"
            }
            if element in number_words and number_words[element] in text:
                return True
        
        # Handle common variations
        variations = {
            "ai": ["artificial intelligence", "a.i."],
            "api": ["application programming interface"],
            "rest": ["restful"],
        }
        
        if element in variations:
            for var in variations[element]:
                if var in text:
                    return True
        
        return False
    
    def _calculate_total_score(
        self,
        exact_match: bool,
        semantic_score: float,
        elements_found: List[str],
        elements_missing: List[str],
        test_case: TestCase
    ) -> float:
        """Calculate total score out of 100"""
        
        # Exact match gets full score
        if exact_match:
            return 100.0
        
        # Otherwise, weighted scoring
        scores = {
            'semantic': semantic_score * 100 * 0.5,  # 50% weight
            'elements': (len(elements_found) / len(test_case.expected_elements)) * 100 * 0.3,  # 30% weight
            'structure': self._check_structural_match(test_case) * 100 * 0.2  # 20% weight
        }
        
        # Difficulty multiplier
        difficulty_multiplier = {
            'easy': 1.0,
            'medium': 0.95,
            'hard': 0.9,
            'adversarial': 0.85
        }
        
        base_score = sum(scores.values())
        multiplier = difficulty_multiplier.get(test_case.difficulty, 1.0)
        
        return min(base_score * multiplier, 100.0)
    
    def _check_structural_match(self, test_case: TestCase) -> float:
        """Check if structural requirements are met"""
        # This is simplified - in reality would check output structure
        # against what the prompt would generate
        if test_case.category in ['format', 'multi-constraint']:
            return 0.8  # Assume reasonable structural match
        return 1.0
    
    def _check_confidence_calibration(
        self, 
        predicted_confidence: float,
        actual_accuracy: float
    ) -> float:
        """Check how well calibrated the confidence is"""
        # Perfect calibration: confidence matches accuracy
        difference = abs(predicted_confidence - actual_accuracy)
        return 1.0 - difference
    
    def _generate_report(self, results: List[EvaluationScore]) -> EvaluationReport:
        """Generate comprehensive evaluation report"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Calculate average score
        average_score = sum(r.total_score for r in results) / total_tests
        
        # Group by category
        scores_by_category = {}
        category_counts = {}
        for result in results:
            test_case = next(tc for tc in get_test_cases() if tc.id == result.test_id)
            category = test_case.category
            
            if category not in scores_by_category:
                scores_by_category[category] = 0
                category_counts[category] = 0
            
            scores_by_category[category] += result.total_score
            category_counts[category] += 1
        
        # Average by category
        for category in scores_by_category:
            scores_by_category[category] /= category_counts[category]
        
        # Group by difficulty
        scores_by_difficulty = {}
        difficulty_counts = {}
        for result in results:
            test_case = next(tc for tc in get_test_cases() if tc.id == result.test_id)
            difficulty = test_case.difficulty
            
            if difficulty not in scores_by_difficulty:
                scores_by_difficulty[difficulty] = 0
                difficulty_counts[difficulty] = 0
            
            scores_by_difficulty[difficulty] += result.total_score
            difficulty_counts[difficulty] += 1
        
        # Average by difficulty
        for difficulty in scores_by_difficulty:
            scores_by_difficulty[difficulty] /= difficulty_counts[difficulty]
        
        # Generate summary
        summary = f"""Evaluation Summary:
- Total Tests: {total_tests}
- Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)
- Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)
- Average Score: {average_score:.1f}/100

Performance by Category:
{chr(10).join(f'  - {cat}: {score:.1f}/100' for cat, score in scores_by_category.items())}

Performance by Difficulty:
{chr(10).join(f'  - {diff}: {score:.1f}/100' for diff, score in scores_by_difficulty.items())}
"""
        
        return EvaluationReport(
            timestamp=datetime.utcnow(),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            average_score=average_score,
            scores_by_category=scores_by_category,
            scores_by_difficulty=scores_by_difficulty,
            individual_results=results,
            summary=summary
        )
    
    async def _save_report(self, report: EvaluationReport):
        """Save evaluation report to file"""
        filename = f"evaluation_report_{report.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to dict
        report_dict = {
            'timestamp': report.timestamp.isoformat(),
            'summary': report.summary,
            'metrics': {
                'total_tests': report.total_tests,
                'passed_tests': report.passed_tests,
                'failed_tests': report.failed_tests,
                'average_score': report.average_score,
                'scores_by_category': report.scores_by_category,
                'scores_by_difficulty': report.scores_by_difficulty
            },
            'individual_results': [asdict(r) for r in report.individual_results]
        }
        
        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"\nReport saved to: {filename}")

async def run_evaluation():
    """Main evaluation runner"""
    async with PromptEvaluator() as evaluator:
        report = await evaluator.evaluate_all()
        print("\n" + "="*50)
        print(report.summary)
        print("="*50)

if __name__ == "__main__":
    asyncio.run(run_evaluation())