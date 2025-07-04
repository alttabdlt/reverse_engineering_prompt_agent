# Reverse Prompt Engineering Detective 🔍

An intelligent FastAPI service that performs **reverse prompt engineering** - analyzing AI-generated outputs to deduce the original prompts that created them.

## 🎯 What is Reverse Prompt Engineering?

Traditional AI flow: **Prompt → AI Model → Output**

This project reverses it: **Output → Detective Agent → Original Prompt**

Given only the AI-generated text, our detective agent attempts to reconstruct what prompt was used to create it. This is like forensic analysis for AI content!

### Example:
- **You see**: "Roses are red, Violets are blue, Sugar is sweet, And so are you"
- **Detective deduces**: "Write a romantic poem"

## 🚀 Quick Start - Interactive Demo

Want to see it in action? Try the interactive demo:

```bash
# Terminal 1: Start the API
cd part1
python start_server.py

# Terminal 2: Run interactive demo
python interactive_demo.py
```

The demo will:
1. Ask you for a prompt
2. Generate output using Gemini
3. Send ONLY the output to the detective
4. Show you if it can figure out your original prompt!

## 🔧 Key Features

- **Multi-Tool Integration**: 
  - Pattern Analyzer: Extracts linguistic and structural patterns
  - Gemini (via Vertex AI): Generates prompt hypotheses
  - Cohere: Validates hypotheses through semantic similarity
  - Enhanced Validator: 6-dimensional scoring system
  
- **Progressive Refinement**: Up to 5 passes, each learning from previous attempts
  
- **Comprehensive Scoring**: Evaluates semantic similarity, structural match, constraint satisfaction, style match, intent preservation, and complexity
  
- **Production-Ready**: FastAPI, Pydantic models, 35+ unit tests, Docker deployment
  
- **Evaluation Framework**: 8 test cases covering simple to adversarial scenarios

## 📋 Prerequisites

- Python 3.10+
- Google Cloud Project with Vertex AI enabled
- Cohere API key
- Docker (for containerized deployment)

## 🛠 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd assessment
```

2. **Set up Python environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure credentials:**

#### Option 1: Interactive Setup (Recommended)
```bash
python setup_credentials.py
```

#### Option 2: Manual Setup
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GOOGLE_CLOUD_PROJECT=your-project-id
VERTEX_AI_LOCATION=us-central1
COHERE_API_KEY=your-cohere-api-key
```

#### Getting Service Account Key
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Navigate to "IAM & Admin" → "Service Accounts"
3. Create key: "Keys" → "Add Key" → "Create new key" → JSON
4. Grant `Vertex AI User` role

#### Alternative: Use JSON Content Directly
```bash
export GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat /path/to/service-account.json)"
```

## 🏃 Running the Service

### API Server
```bash
cd part1
python start_server.py
```

Access at:
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

### Interactive Demo
```bash
python interactive_demo.py
```

**How it works:**
1. You enter a prompt (e.g., "Write a haiku about coding")
2. Gemini generates output from your prompt
3. ONLY the output is sent to the detective (not your prompt!)
4. The detective tries to reverse engineer the original prompt
5. Compare results to see accuracy

**Try these prompts:**
- Simple: "List 5 benefits of exercise"
- Creative: "Write a haiku about the ocean"
- Technical: "Explain recursion in simple terms"
- Complex: "As a pirate, describe your favorite treasure in exactly 3 sentences"

### Advanced Showcase
```bash
python showcase_advanced.py
```

Tests edge cases and complex scenarios.

### Running Tests
```bash
cd part1
pytest -v                     # Run all tests
pytest --cov=. --cov-report=html  # With coverage
python run_evaluation.py      # Run evaluation suite
```

## 📁 Project Structure

```
assessment/
├── part1/                      # Core implementation
│   ├── main.py                 # FastAPI application
│   ├── agents/                 # Agent implementations
│   │   └── prompt_detective.py # Main detective agent
│   ├── tools/                  # Tool integrations (4 tools)
│   │   ├── pattern_analyzer.py # Pattern extraction
│   │   ├── hypothesis_generator.py # Gemini integration
│   │   ├── validator.py        # Cohere validation
│   │   └── enhanced_validator.py # Advanced validation
│   ├── models/                 # Pydantic models
│   ├── evaluation/             # Test framework
│   │   └── test_cases.py       # 8 test scenarios
│   └── tests/                  # 35+ unit tests
├── part2/                      # Deployment
│   ├── Dockerfile              # Multi-stage build
│   └── deploy.sh               # Deployment scripts
├── part3/                      # Conceptual answers
│   └── ANSWERS.md              # System design Q&A
├── interactive_demo.py         # Try it yourself!
└── showcase_advanced.py        # Advanced test cases
```

## 🧪 Testing

### Unit Tests
```bash
cd part1
pytest -v
```

35+ tests covering:
- Input validation
- Tool routing
- System prompt effects
- API endpoints
- Evaluation logic

### Evaluation Framework
```bash
cd part1
python run_evaluation.py
```

Runs 8 comprehensive test cases:
1. Simple instructions
2. Formatted lists
3. Role-based outputs
4. Technical explanations
5. Creative writing
6. Code generation
7. Multi-constraint scenarios
8. Edge cases

## 🐳 Docker Deployment

```bash
cd part2
docker build -t prompt-detective .
docker run -p 8000:8000 \
  -e GOOGLE_APPLICATION_CREDENTIALS_JSON='<json-content>' \
  -e COHERE_API_KEY='<your-key>' \
  prompt-detective
```

## 🔍 How It Works

### 1. Pattern Analysis
Extracts features like:
- Text structure (lists, paragraphs, code blocks)
- Linguistic patterns (formal/informal, technical terms)
- Constraints (word count, format requirements)

### 2. Hypothesis Generation
Uses Gemini to generate 3-5 likely prompts based on patterns.

### 3. Validation & Scoring
Each hypothesis is validated by:
- Generating output from the hypothesized prompt
- Comparing with original using 6 scoring dimensions
- Calculating confidence scores

### 4. Progressive Refinement
Up to 5 passes, each improving based on:
- Missing elements identified
- Validation feedback
- Confidence thresholds

## 📊 API Usage

```bash
POST /analyze
```

Request:
```json
{
  "output_text": "1. Exercise\n2. Eat healthy\n3. Sleep well",
  "max_attempts": 5,
  "context": "health advice"
}
```

Response:
```json
{
  "success": true,
  "result": {
    "best_hypothesis": {
      "prompt": "List 3 tips for healthy living",
      "confidence": 0.89,
      "reasoning": "Numbered list format with health-related items"
    },
    "confidence": "high",
    "attempts_used": 2
  }
}
```

## 🚨 Recent Updates

- **Fixed environment loading issues** (June 30, 2024)
  - .env now loads correctly when running from part1/ directory
  - Created symlink to ensure environment variables are found
  - Fixed ValidationResult model compatibility with enhanced scoring
- **Fixed Cohere API compatibility** for v5.15.0
  - Removed invalid model parameter from legacy generate()
  - Added 30s timeout to prevent long waits
  - Fixed embed API warnings by using proper instance checking for Client type detection
  - See [COHERE_TROUBLESHOOTING.md](COHERE_TROUBLESHOOTING.md) for details
- **Added interactive demo** for hands-on testing
- **Improved error handling** and fallback mechanisms
- **Enhanced validation** with both ClientV2 and legacy Client support

## ⚠️ Known Limitations

### Over-fitting in Prompt Reconstruction
The current system tends to over-specify prompts, assuming every output characteristic was explicitly requested. For example:
- **Actual prompt**: "Explain how backpropagation works"
- **System reconstructs**: Complex 45-word prompt with specific constraints
- **Result**: Low accuracy on simple prompts

**Solution in development**: Implementing Occam's Razor principle with simplicity scoring. See [CLAUDE.md](CLAUDE.md#known-issues--solutions) for implementation details and improvement plan.

## 🤝 Assessment Fulfillment

This project meets all technical assessment requirements:

### Part 1 ✅
- FastAPI service with agent architecture
- Multiple tools (4 integrated: Pattern Analyzer, Gemini, Cohere, Enhanced Validator)
- System prompt guiding agent behavior
- 8 evaluation test cases
- Automated evaluation framework
- Comprehensive pytest suite (35+ tests)
- Pydantic models throughout

### Part 2 ✅
- Docker containerization
- Multi-stage build optimization
- Deployment scripts
- Resource configuration

### Part 3 ✅
- Conceptual answers on agent evaluation
- Prompt engineering methodology
- System design documentation
- See [part3/ANSWERS.md](part3/ANSWERS.md) for detailed responses

## 📝 License

This project is part of a technical assessment.

---

**Try the interactive demo to see reverse prompt engineering in action!** 🎯