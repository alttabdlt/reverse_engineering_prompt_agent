# Interactive Reverse Prompt Engineering Demo

This interactive demo showcases the core concept of the project: **reverse engineering prompts from AI-generated outputs**.

## How It Works

1. **You enter a prompt** (e.g., "Write a haiku about coding")
2. **Gemini generates output** from your prompt
3. **ONLY the output is sent to the detective** (not your original prompt!)
4. **The detective tries to reverse engineer** what prompt created that output
5. **Compare the results** to see how well it worked

## Running the Demo

1. **Start the API server** (if not already running):
   ```bash
   cd part1
   python start_server.py
   ```

2. **Run the interactive demo**:
   ```bash
   python interactive_demo.py
   ```

3. **Try different prompts** to test the detective's capabilities:
   - Simple: "List 5 benefits of exercise"
   - Creative: "Write a haiku about the ocean"
   - Technical: "Explain recursion in simple terms"
   - Complex: "As a pirate, describe your favorite treasure in exactly 3 sentences"

## Example Session

```
🔮 Interactive Reverse Prompt Engineering Demo
============================================================

Enter a prompt to test reverse engineering:
Your prompt: Write a haiku about coding

[Generating output...]
[Detective analyzing...]

┌─ 1️⃣ Your Original Prompt ─┐
│ Write a haiku about coding │
└────────────────────────────┘

┌─ 2️⃣ Generated Output ─────┐
│ Bugs hide in silence        │
│ Logic flows through my mind │  
│ Code springs to life        │
└────────────────────────────┘

┌─ 3️⃣ Detective's Result ────┐
│ Write a haiku about         │
│ programming or coding       │
└────────────────────────────┘

Confidence: 92%
✅ EXCELLENT MATCH!
```

## What This Demonstrates

1. **Pattern Recognition**: The detective identifies structural patterns (5-7-5 syllable haiku format)
2. **Content Analysis**: It recognizes technical terminology related to coding
3. **Intent Inference**: It deduces the user wanted creative output in a specific format
4. **Multi-Tool Integration**: Uses Gemini for hypothesis generation and Cohere for validation

## Success Metrics

- **Excellent Match** (>85% confidence): Detective accurately reconstructs the prompt
- **Good Match** (70-85%): Captures the essence but may differ in phrasing
- **Partial Match** (<70%): Identifies the type of request but misses specifics

This demonstrates the practical application of reverse prompt engineering - a unique AI capability that could be useful for:
- Understanding AI-generated content origin
- Debugging AI outputs
- Educational tools for prompt engineering
- Content attribution and analysis