#!/usr/bin/env python3
"""
Advanced showcase demonstrating edge cases and complex scenarios.
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any
import time

ADVANCED_CASES = [
    {
        "name": "Ambiguous Output",
        "output": "The process involves multiple steps and careful consideration of various factors.",
        "expected_challenge": "Generic output that could match many prompts",
        "difficulty": "very hard"
    },
    {
        "name": "Mixed Constraints",
        "output": """As a data scientist, here are my top 3 recommendations:

1. **Clean your data** - This is crucial because garbage in means garbage out
2. **Choose the right algorithm** - Don't use a sledgehammer to crack a nut
3. **Validate thoroughly** - Always use cross-validation to avoid overfitting

Remember: "In God we trust, all others must bring data!" 📊""",
        "expected_challenge": "Multiple constraints: persona, format, style, emoji",
        "difficulty": "expert"
    },
    {
        "name": "Adversarial Case",
        "output": """Write a Python function that calculates the factorial of a number:

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)""",
        "expected_challenge": "Output contains what looks like a prompt itself",
        "difficulty": "expert"
    },
    {
        "name": "Non-AI Text",
        "output": """Breaking News: Local bakery wins award for best croissants in the city. The owner, Marie Dubois, credits her success to using traditional French techniques passed down through three generations. "We wake up at 3 AM every day to ensure fresh pastries," she told reporters.""",
        "expected_challenge": "Might be human-written news, not AI-generated",
        "difficulty": "hard"
    },
    {
        "name": "Complex Technical with Examples",
        "output": """A monad is a design pattern that provides a way to wrap values and compose functions that return wrapped values. Here's a simple Maybe monad in Python:

```python
class Maybe:
    def __init__(self, value):
        self.value = value
    
    def bind(self, func):
        if self.value is None:
            return Maybe(None)
        return func(self.value)
    
    def __repr__(self):
        return f"Maybe({self.value})"

# Example usage:
result = Maybe(5).bind(lambda x: Maybe(x * 2)).bind(lambda x: Maybe(x + 1))
print(result)  # Maybe(11)
```

This pattern is particularly useful for handling nullable values without explicit null checks.""",
        "expected_challenge": "Code examples within explanation, technical depth",
        "difficulty": "expert"
    }
]

async def test_advanced_case(session: aiohttp.ClientSession, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Test an advanced case with detailed analysis."""
    url = "http://localhost:8000/analyze"
    
    print(f"\n{'='*70}")
    print(f"🧪 Advanced Test: {test_case['name']}")
    print(f"Difficulty: {test_case['difficulty'].upper()}")
    print(f"Challenge: {test_case['expected_challenge']}")
    print(f"{'='*70}")
    
    # Show preview of output
    preview = test_case['output'][:150]
    if len(test_case['output']) > 150:
        preview += "..."
    print(f"\nOutput preview:\n{preview}")
    
    start_time = time.time()
    
    try:
        payload = {
            "output_text": test_case["output"],
            "max_attempts": 5,
            "context": {"test_type": "advanced", "difficulty": test_case["difficulty"]}
        }
        
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as response:
            result = await response.json()
            
        elapsed_time = time.time() - start_time
        
        if result.get("success"):
            detection = result["result"]
            hypothesis = detection["best_hypothesis"]
            
            print(f"\n✅ Analysis Complete")
            print(f"\n🎯 Detected Prompt:\n{hypothesis['prompt']}")
            print(f"\n📊 Confidence Score: {hypothesis['confidence']:.2%} ({detection['confidence']})")
            print(f"\n🔑 Key Elements Identified:")
            for elem in hypothesis['key_elements']:
                print(f"  • {elem}")
            
            print(f"\n💭 Reasoning:\n{hypothesis['reasoning']}")
            
            # Enhanced scoring details
            if detection.get("enhanced_scoring"):
                scoring = detection["enhanced_scoring"]
                print(f"\n📈 Detailed Scoring Breakdown:")
                print(f"  • Semantic similarity: {scoring['semantic_similarity']:.2%}")
                print(f"  • Structural match: {scoring['structural_match']:.2%}")
                print(f"  • Constraint satisfaction: {scoring['constraint_satisfaction']:.2%}")
                print(f"  • Style match: {scoring['style_match']:.2%}")
                print(f"  • Intent preservation: {scoring['intent_preservation']:.2%}")
                print(f"  • Complexity penalty: {scoring['complexity_penalty']:.2f}")
                
                conf_low, conf_high = scoring['confidence_interval']
                print(f"  • Confidence interval: [{conf_low:.2%}, {conf_high:.2%}]")
            
            # Comparison details
            if detection.get("comparison"):
                comp = detection["comparison"]
                print(f"\n🔍 Validation Comparison:")
                
                if comp['similarities']:
                    print(f"\n  ✓ Similarities ({len(comp['similarities'])}):")
                    for sim in comp['similarities'][:3]:
                        print(f"    - {sim}")
                
                if comp['differences']:
                    print(f"\n  ✗ Differences ({len(comp['differences'])}):")
                    for diff in comp['differences'][:3]:
                        print(f"    - {diff}")
            
            print(f"\n⏱️ Processing time: {elapsed_time:.2f}s")
            print(f"🔄 Attempts used: {detection['attempts_used']}")
            
            return {
                "success": True,
                "test_case": test_case["name"],
                "confidence": hypothesis["confidence"],
                "processing_time": elapsed_time,
                "attempts": detection["attempts_used"]
            }
            
        else:
            print(f"\n❌ Analysis failed: {result.get('message', 'Unknown error')}")
            return {
                "success": False,
                "test_case": test_case["name"],
                "error": result.get("message", "Unknown error"),
                "processing_time": elapsed_time
            }
            
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        return {
            "success": False,
            "test_case": test_case["name"],
            "error": str(e),
            "processing_time": time.time() - start_time
        }

async def run_advanced_showcase():
    """Run the advanced showcase with edge cases."""
    print("🔬 Advanced Showcase - Edge Cases and Complex Scenarios")
    print("="*70)
    
    # Check server
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get("http://localhost:8000/health") as response:
                if response.status != 200:
                    print("❌ Server not running. Start with: cd part1 && python start_server.py")
                    return
        except:
            print("❌ Cannot connect to server. Start with: cd part1 && python start_server.py")
            return
    
    # Run advanced tests
    results = []
    async with aiohttp.ClientSession() as session:
        for test_case in ADVANCED_CASES:
            result = await test_advanced_case(session, test_case)
            results.append(result)
            await asyncio.sleep(2)  # Longer delay for complex cases
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 ADVANCED SHOWCASE SUMMARY")
    print(f"{'='*70}")
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"\nTotal advanced cases: {len(results)}")
    print(f"✅ Successfully analyzed: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        avg_confidence = sum(r["confidence"] for r in successful) / len(successful)
        avg_time = sum(r["processing_time"] for r in successful) / len(successful)
        avg_attempts = sum(r["attempts"] for r in successful) / len(successful)
        
        print(f"\n📈 Performance Metrics:")
        print(f"  • Average confidence: {avg_confidence:.2%}")
        print(f"  • Average processing time: {avg_time:.2f}s")
        print(f"  • Average attempts needed: {avg_attempts:.1f}")
    
    print("\n🏆 Challenge Results:")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"\n{status} {r['test_case']}:")
        if r["success"]:
            conf_level = "High" if r["confidence"] > 0.8 else "Medium" if r["confidence"] > 0.6 else "Low"
            print(f"   Confidence: {r['confidence']:.2%} ({conf_level})")
            print(f"   Time: {r['processing_time']:.2f}s | Attempts: {r['attempts']}")
        else:
            print(f"   Error: {r['error']}")

if __name__ == "__main__":
    asyncio.run(run_advanced_showcase())