# Part 1: Implementation & Prompt Evaluation

This directory contains the core implementation of the Reverse Prompt Engineering Detective API.

## Quick Start

1. **Start the server:**
   ```bash
   python start_server.py
   ```

2. **Access the API:**
   - Base URL: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## Project Structure

```
part1/
├── main.py              # FastAPI application entry point
├── agents/              # Agent implementations
│   └── prompt_detective.py  # Main reverse engineering agent
├── tools/               # Tool integrations (2+ tools as required)
│   ├── pattern_analyzer.py      # Tool 1: Pattern extraction
│   ├── hypothesis_generator.py  # Tool 2: Gemini integration
│   ├── validator.py             # Tool 3: Cohere validation
│   └── enhanced_validator.py    # Tool 4: Advanced validation
├── models/              # Pydantic models
│   ├── requests.py      # API request schemas
│   ├── responses.py     # API response schemas
│   ├── analysis.py      # Pattern analysis models
│   ├── hypothesis.py    # Hypothesis models
│   └── validation.py    # Validation models
├── evaluation/          # Evaluation framework
│   ├── evaluator.py     # Automated evaluation engine
│   └── test_cases.py    # 8 comprehensive test cases
├── tests/               # Unit tests
│   ├── test_api.py      # API endpoint tests
│   ├── test_agent.py    # Agent behavior tests
│   ├── test_models.py   # Model validation tests
│   └── test_pattern_analyzer.py  # Tool tests
└── pytest.ini           # Pytest configuration
```

## Running Tests

### Unit Tests (35 tests covering all requirements)
```bash
pytest -v
```

### Test Coverage
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

### Evaluation Framework
```bash
python run_evaluation.py
```

This runs 8 test cases covering:
- Basic instructions
- Formatted outputs
- Role-based prompts
- Technical content
- Creative writing
- Code generation
- Multi-constraint scenarios
- Edge cases

## API Endpoints

### POST /analyze
Analyzes AI-generated output to reconstruct the original prompt.

**Request:**
```json
{
  "output_text": "AI generated text to analyze",
  "max_attempts": 5,
  "context": {
    "domain": "technical",
    "complexity": "high"
  }
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "best_hypothesis": {
      "prompt": "Reconstructed prompt",
      "confidence": 0.85,
      "reasoning": "Detailed explanation",
      "key_elements": ["element1", "element2"]
    },
    "confidence": "high",
    "attempts_used": 3
  }
}
```

### GET /health
Health check endpoint for monitoring.

### GET /metrics
Basic performance metrics.

## System Prompt

The agent uses a carefully crafted system prompt that:
- Defines the agent as a "reverse prompt engineering detective"
- Guides systematic analysis through 5 potential passes
- Enforces structured JSON output
- Handles edge cases and ambiguous outputs
- Maintains professional, analytical tone

See `agents/prompt_detective.py` for the full system prompt implementation.

## Tool Usage

The agent utilizes multiple tools as required:

1. **Pattern Analyzer**: Extracts linguistic and structural patterns
2. **Hypothesis Generator**: Uses Google's Gemini 1.5 Flash for prompt generation
3. **Validator**: Uses Cohere for semantic similarity validation
4. **Enhanced Validator**: Provides 6-dimensional scoring analysis

Each tool is properly integrated with error handling and fallback mechanisms.

## Development

### Adding New Test Cases
Edit `evaluation/test_cases.py` to add new scenarios.

### Modifying Agent Behavior
Update the system prompt in `agents/prompt_detective.py`.

### Running in Debug Mode
```bash
export LOG_LEVEL=DEBUG
python start_server.py
```