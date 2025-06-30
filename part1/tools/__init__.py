from .base import Tool
from .pattern_analyzer import PatternAnalyzer
from .hypothesis_generator import GeminiHypothesisGenerator
from .validator import CohereValidator

# V2 improvements (not yet integrated) - uncomment when ready to use
# from .prompt_simplicity_scorer import PromptSimplicityScorer
# from .progressive_hint_engine import ProgressiveHintEngine

__all__ = [
    "Tool",
    "PatternAnalyzer",
    "GeminiHypothesisGenerator",
    "CohereValidator"
    # Future V2 tools:
    # "PromptSimplicityScorer",
    # "ProgressiveHintEngine"
]