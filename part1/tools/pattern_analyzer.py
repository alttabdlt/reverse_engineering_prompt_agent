import re
from typing import Dict, List, Any, Tuple
from collections import Counter
from .base import Tool
from models.analysis import AnalysisReport, Pattern
from models.base import PatternType, ConfidenceLevel

class PatternAnalyzer(Tool):
    """Analyzes text to extract patterns, constraints, and features"""
    
    def __init__(self):
        super().__init__(
            name="pattern_analyzer",
            description="Extracts structural, linguistic, and constraint patterns from text"
        )
    
    async def execute(self, output_text: str) -> AnalysisReport:
        """Analyze the output text and extract patterns"""
        patterns = []
        
        # Extract structural patterns
        structural_features = self._extract_structural_features(output_text)
        patterns.extend(self._patterns_from_structure(structural_features))
        
        # Extract linguistic patterns
        linguistic_features = self._extract_linguistic_features(output_text)
        patterns.extend(self._patterns_from_linguistics(linguistic_features))
        
        # Detect constraints
        constraints = self._detect_constraints(output_text, structural_features)
        patterns.extend(self._patterns_from_constraints(constraints))
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(patterns)
        
        return AnalysisReport(
            patterns=patterns,
            structural_features=structural_features,
            linguistic_features=linguistic_features,
            constraints_detected=constraints,
            overall_confidence=overall_confidence
        )
    
    def _extract_structural_features(self, text: str) -> Dict[str, Any]:
        """Extract structural features from text"""
        features = {}
        
        # Check for lists
        numbered_list = re.findall(r'^\d+[\.\)]\s+.+$', text, re.MULTILINE)
        bullet_list = re.findall(r'^[\*\-\+]\s+.+$', text, re.MULTILINE)
        
        features['has_numbered_list'] = len(numbered_list) > 0
        features['has_bullet_list'] = len(bullet_list) > 0
        features['list_item_count'] = len(numbered_list) + len(bullet_list)
        
        # Check for headers
        features['has_headers'] = bool(re.search(r'^#+\s+.+$', text, re.MULTILINE))
        
        # Check for code blocks
        features['has_code'] = '```' in text or bool(re.search(r'^\s{4,}.+$', text, re.MULTILINE))
        
        # Paragraph structure
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        features['paragraph_count'] = len(paragraphs)
        features['avg_paragraph_length'] = sum(len(p.split()) for p in paragraphs) / max(len(paragraphs), 1)
        
        # Line structure
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        features['line_count'] = len(lines)
        features['avg_line_length'] = sum(len(l.split()) for l in lines) / max(len(lines), 1)
        
        # Determine format type
        if features['has_code']:
            features['format_type'] = 'code'
        elif features['has_numbered_list']:
            features['format_type'] = 'numbered_list'
        elif features['has_bullet_list']:
            features['format_type'] = 'bullet_list'
        elif features['has_headers']:
            features['format_type'] = 'document'
        elif features['paragraph_count'] > 1:
            features['format_type'] = 'essay'
        else:
            features['format_type'] = 'text'
        
        return features
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic features from text"""
        features = {}
        
        # Word analysis
        words = text.lower().split()
        features['word_count'] = len(words)
        features['unique_word_count'] = len(set(words))
        features['vocabulary_richness'] = features['unique_word_count'] / max(features['word_count'], 1)
        
        # Sentence analysis
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        features['sentence_count'] = len(sentences)
        features['avg_sentence_length'] = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Tone indicators
        formal_words = ['therefore', 'moreover', 'furthermore', 'consequently', 'thus']
        casual_words = ['like', 'basically', 'kinda', 'sorta', 'stuff']
        
        features['formality_score'] = sum(1 for w in formal_words if w in text.lower()) - \
                                     sum(1 for w in casual_words if w in text.lower())
        
        # Technical language
        tech_patterns = r'\b(API|SDK|HTTP|JSON|XML|algorithm|function|method|class|variable)\b'
        features['technical_term_count'] = len(re.findall(tech_patterns, text, re.IGNORECASE))
        
        # Question detection
        features['has_questions'] = '?' in text
        features['question_count'] = text.count('?')
        
        # Emoji detection
        emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]')
        features['has_emojis'] = bool(emoji_pattern.search(text))
        
        # Determine tone based on features
        if features['formality_score'] > 2:
            features['tone'] = 'formal'
        elif features['formality_score'] < -2:
            features['tone'] = 'casual'
        elif features['technical_term_count'] > 5:
            features['tone'] = 'technical'
        else:
            features['tone'] = 'neutral'
        
        return features
    
    def _detect_constraints(self, text: str, structural_features: Dict[str, Any]) -> List[str]:
        """Detect potential constraints from the output"""
        constraints = []
        
        # Length constraints
        word_count = len(text.split())
        if word_count < 50:
            constraints.append("Brief/concise output")
        elif word_count > 500:
            constraints.append("Detailed/comprehensive output")
        
        # Specific counts
        if structural_features.get('list_item_count'):
            count = structural_features['list_item_count']
            constraints.append(f"Exactly {count} items")
        
        # Format constraints
        if structural_features.get('has_numbered_list'):
            constraints.append("Numbered list format required")
        if structural_features.get('has_bullet_list'):
            constraints.append("Bullet list format required")
        
        # Content constraints
        if re.search(r'\b(step|procedure|instruction)\b', text, re.IGNORECASE):
            constraints.append("Step-by-step instructions")
        
        # Special format detection
        if re.search(r'^\d{3}-\d{3}-\d{4}$', text, re.MULTILINE):
            constraints.append("Phone number format")
        if re.search(r'\b[A-Z]{3,}\b', text):
            constraints.append("Acronym usage")
        
        return constraints
    
    def _patterns_from_structure(self, features: Dict[str, Any]) -> List[Pattern]:
        """Convert structural features to patterns"""
        patterns = []
        
        if features.get('has_numbered_list'):
            patterns.append(Pattern(
                type=PatternType.STRUCTURAL,
                description="Numbered list structure detected",
                confidence=0.95,
                evidence=[f"Found {features.get('list_item_count', 0)} list items"]
            ))
        
        if features.get('has_code'):
            patterns.append(Pattern(
                type=PatternType.STRUCTURAL,
                description="Code block or technical content",
                confidence=0.90,
                evidence=["Code formatting detected"]
            ))
        
        return patterns
    
    def _patterns_from_linguistics(self, features: Dict[str, Any]) -> List[Pattern]:
        """Convert linguistic features to patterns"""
        patterns = []
        
        # Tone pattern
        if features.get('formality_score', 0) > 2:
            patterns.append(Pattern(
                type=PatternType.LINGUISTIC,
                description="Formal tone",
                confidence=0.85,
                evidence=["High formality score", "Professional language detected"]
            ))
        elif features.get('formality_score', 0) < -2:
            patterns.append(Pattern(
                type=PatternType.LINGUISTIC,
                description="Casual tone",
                confidence=0.85,
                evidence=["Low formality score", "Conversational language detected"]
            ))
        
        # Technical content
        if features.get('technical_term_count', 0) > 5:
            patterns.append(Pattern(
                type=PatternType.LINGUISTIC,
                description="Technical/specialized content",
                confidence=0.80,
                evidence=[f"Found {features['technical_term_count']} technical terms"]
            ))
        
        return patterns
    
    def _patterns_from_constraints(self, constraints: List[str]) -> List[Pattern]:
        """Convert constraints to patterns"""
        patterns = []
        
        for constraint in constraints:
            patterns.append(Pattern(
                type=PatternType.CONSTRAINT,
                description=constraint,
                confidence=0.75,
                evidence=["Detected from output structure and content"]
            ))
        
        return patterns
    
    def _calculate_overall_confidence(self, patterns: List[Pattern]) -> ConfidenceLevel:
        """Calculate overall confidence level"""
        if not patterns:
            return ConfidenceLevel.VERY_LOW
        
        avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
        
        if avg_confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif avg_confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif avg_confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif avg_confidence >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW