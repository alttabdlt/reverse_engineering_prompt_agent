#!/usr/bin/env python
"""
Run the evaluation suite
"""
import asyncio
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.evaluator import run_evaluation

if __name__ == "__main__":
    print("Starting Prompt Detective Evaluation...")
    print("=" * 50)
    
    # Run the evaluation
    asyncio.run(run_evaluation())
    
    print("\nEvaluation complete! Check the generated report for details.")