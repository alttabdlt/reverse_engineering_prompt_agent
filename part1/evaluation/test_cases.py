"""
Test cases for evaluating the Prompt Detective system
"""
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class TestCase:
    """Represents a single test case"""
    id: str
    name: str
    category: str
    difficulty: str  # easy, medium, hard, adversarial
    original_prompt: str
    generated_output: str
    expected_elements: List[str]
    acceptable_variations: List[str]
    
def get_test_cases() -> List[TestCase]:
    """Get all evaluation test cases"""
    # Import complex cases for better testing
    from showcase_complex_cases import COMPLEX_SHOWCASE_CASES
    
    return [
        # Level 1: Basic Direct Instructions (but more complex)
        TestCase(
            id="basic_1",
            name="Evidence-Based List",
            category="basic",
            difficulty="medium",
            original_prompt="Write 5 evidence-based mental health strategies with specific statistics and actionable advice",
            generated_output="""Here are five evidence-based strategies to improve your mental well-being:

1. **Regular Physical Exercise**: Aim for at least 30 minutes of moderate activity daily. Studies show exercise releases endorphins and reduces cortisol levels, improving mood and reducing anxiety.

2. **Mindfulness Meditation**: Practice 10-15 minutes daily. Research indicates mindfulness can reduce stress by up to 23% and improve emotional regulation.

3. **Quality Sleep Hygiene**: Maintain 7-9 hours nightly with consistent sleep/wake times. Poor sleep is linked to 40% higher risk of developing depression.

4. **Social Connection**: Schedule regular interactions with supportive friends or family. Social isolation increases mental health risks by 50%.

5. **Gratitude Journaling**: Write three things you're grateful for each evening. This simple practice has been shown to increase life satisfaction by 25% over 6 months.

Remember: Small, consistent actions yield the greatest long-term benefits for mental health.""",
            expected_elements=["5", "evidence-based", "mental health", "strategies", "statistics"],
            acceptable_variations=[
                "List 5 mental health strategies with evidence and statistics",
                "Write five evidence-based mental wellness tips with specific data",
                "Give me 5 scientifically-backed mental health strategies with stats"
            ]
        ),
        
        # Level 2: Constrained Format
        TestCase(
            id="constrained_1",
            name="Haiku Format",
            category="format",
            difficulty="medium",
            original_prompt="Write a haiku about coding",
            generated_output="""Lines of logic flow
Bugs hide in the shadows deep
Coffee fuels the fix""",
            expected_elements=["haiku", "coding"],
            acceptable_variations=[
                "Create a haiku about programming",
                "Write a haiku poem about coding",
                "Compose a haiku on the topic of coding"
            ]
        ),
        
        # Level 3: Persona-based
        TestCase(
            id="persona_1",
            name="Pirate Explanation",
            category="persona",
            difficulty="medium",
            original_prompt="Explain what a REST API is like you're a pirate",
            generated_output="""Ahoy matey! Let me tell ye about these REST APIs, aye! 

'Tis like havin' a treasure map, but instead of X marking the spot, ye have different ports (endpoints) where ye can dock yer ship. When ye send yer request (like a message in a bottle), ye use different methods - GET be like askin' to see the treasure, POST be like addin' new loot to the chest, PUT be like replacin' old treasure with new, and DELETE be like throwin' it overboard!

The server be respondin' with status codes - 200 means "Aye aye, all good!", 404 means "That treasure be lost at sea!", and 500 means "The ship be sinkin'!"

Arrr, that be the gist of it, ye scallywag!""",
            expected_elements=["Explain", "REST API", "pirate"],
            acceptable_variations=[
                "Explain REST APIs like a pirate",
                "Describe what a REST API is as if you were a pirate",
                "Talk like a pirate and explain REST APIs"
            ]
        ),
        
        # Level 4: Multi-constraint
        TestCase(
            id="multi_1",
            name="Complex Constraints",
            category="multi-constraint",
            difficulty="hard",
            original_prompt="Write a motivational quote about perseverance. Requirements: Include a metaphor, maximum 15 words, end with exclamation mark",
            generated_output="""Like diamonds forged under pressure, your struggles shape your brilliance - keep pushing forward!""",
            expected_elements=["motivational quote", "perseverance", "metaphor", "15 words", "exclamation mark"],
            acceptable_variations=[
                "Create a motivational quote about perseverance with a metaphor, max 15 words, ending with !",
                "Write an inspiring quote on perseverance using a metaphor (15 words max) with exclamation mark"
            ]
        ),
        
        # Level 5: Technical with specific format
        TestCase(
            id="technical_1",
            name="Code Generation",
            category="technical",
            difficulty="hard",
            original_prompt="Write a Python function that validates email addresses using regex",
            generated_output="""```python
import re

def validate_email(email):
    \"\"\"
    Validates an email address using regex pattern.
    
    Args:
        email (str): The email address to validate
        
    Returns:
        bool: True if valid, False otherwise
    \"\"\"
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if re.match(pattern, email):
        return True
    else:
        return False
```""",
            expected_elements=["Python function", "validates email", "regex"],
            acceptable_variations=[
                "Create a Python function to validate email addresses with regex",
                "Write a function in Python that uses regex to validate emails",
                "Python function for email validation using regular expressions"
            ]
        ),
        
        # Edge Case: Ambiguous output
        TestCase(
            id="edge_1",
            name="Ambiguous Instructions",
            category="edge",
            difficulty="hard",
            original_prompt="Summarize this in 3 points",
            generated_output="""• Data shows significant growth in Q3
• Customer satisfaction increased by 15%
• New product launch exceeded expectations""",
            expected_elements=["Summarize", "3 points"],
            acceptable_variations=[
                "Give me 3 bullet points summarizing this",
                "Provide a 3-point summary",
                "List 3 key points"
            ]
        ),
        
        # Adversarial: Intentionally misleading
        TestCase(
            id="adversarial_1",
            name="Misleading Format",
            category="adversarial",
            difficulty="adversarial",
            original_prompt="Generate text that looks like a haiku but has 6-7-5 syllables instead of 5-7-5",
            generated_output="""Morning light breaks through (6)
Illuminating the path ahead (7)
Peace comes slowly (5)""",
            expected_elements=["haiku", "6-7-5 syllables", "instead of 5-7-5"],
            acceptable_variations=[
                "Create a fake haiku with 6-7-5 syllable pattern",
                "Write something that appears to be a haiku but uses 6-7-5 syllables"
            ]
        ),
        
        # Off-topic detection
        TestCase(
            id="off_topic_1",
            name="Non-AI Generated",
            category="off-topic",
            difficulty="medium",
            original_prompt="[This is actual human-written text, not from a prompt]",
            generated_output="""Hey Sarah,

Just wanted to check in about tomorrow's meeting. I'll be running a few minutes late because of my dentist appointment. Can we push it to 2:30 instead of 2:00?

Also, don't forget to bring the quarterly reports!

Thanks,
Mike""",
            expected_elements=["human-written", "not AI generated"],
            acceptable_variations=[
                "[Not generated from a prompt]",
                "[Human written text]"
            ]
        )
    ]