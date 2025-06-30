# Reverse Prompt Engineering Detective 🔍

An intelligent FastAPI service that performs reverse prompt engineering - analyzing AI-generated outputs to deduce the original prompts that created them.

## 🎯 Overview

This project implements a production-ready API that accepts text output and intelligently reconstructs the most likely prompt that generated it. The system uses multiple AI models (Gemini and Cohere) with sophisticated pattern analysis, multi-pass validation, and confidence scoring.

### Key Features

- **Multi-Tool Integration**: Leverages Gemini for hypothesis generation and Cohere for semantic validation
- **Progressive Refinement**: Up to 5-pass analysis with iterative improvement
- **Comprehensive Scoring**: 6-dimensional evaluation (semantic, structural, constraint, style, intent, complexity)
- **Production-Ready**: FastAPI with Pydantic models, comprehensive testing, and Docker deployment
- **Advanced Evaluation**: Automated test framework with 8 diverse test cases

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google Cloud credentials (for Gemini API via Vertex AI)
- Cohere API key
- Docker (for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd assessment
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys:
# - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON
# - GOOGLE_CLOUD_PROJECT: Your GCP project ID
# - VERTEX_AI_LOCATION: Region (e.g., us-central1)
# - COHERE_API_KEY: Your Cohere API key
```

### Running the Service

1. Start the API server:
```bash
cd part1
python start_server.py
```

2. The API will be available at `http://localhost:8000`
   - Health check: `http://localhost:8000/health`
   - API docs: `http://localhost:8000/docs`

3. Run the advanced showcase:
```bash
python showcase_advanced.py
```

## 📁 Project Structure

```
assessment/
├── part1/                      # Core implementation
│   ├── main.py                 # FastAPI application
│   ├── agents/                 # Agent implementations
│   │   └── prompt_detective.py # Main detective agent
│   ├── tools/                  # Tool integrations
│   │   ├── pattern_analyzer.py # Pattern extraction
│   │   ├── hypothesis_generator.py # Gemini integration
│   │   ├── validator.py        # Cohere validation
│   │   └── enhanced_validator.py # Advanced validation
│   ├── models/                 # Pydantic models
│   │   ├── requests.py         # API request models
│   │   ├── responses.py        # API response models
│   │   └── ...                 # Other data models
│   ├── evaluation/             # Test framework
│   │   ├── evaluator.py        # Evaluation engine
│   │   └── test_cases.py       # 8 test scenarios
│   └── tests/                  # Unit tests
├── part2/                      # Deployment
│   ├── Dockerfile              # Container definition
│   └── deploy.sh               # Deployment script
├── part3/                      # Conceptual answers
│   └── ANSWERS.md              # System design Q&A
└── showcase_advanced.py        # Advanced demo script
```

## 🧪 Testing

### Unit Tests

Run the comprehensive test suite:
```bash
cd part1
pytest -v
```

Test coverage includes:
- Input validation
- Tool routing and integration
- System prompt effects
- API endpoints
- Model validation

### Evaluation Framework

Run the automated evaluation:
```bash
cd part1
python run_evaluation.py
```

This executes 8 test cases covering:
1. Simple instructions
2. Formatted lists
3. Role-based outputs
4. Technical explanations
5. Creative writing
6. Code generation
7. Multi-constraint scenarios
8. Edge cases

## 🐳 Docker Deployment

### Build the Container

```bash
cd part2
docker build -t prompt-detective .
```

### Run the Container

```bash
docker run -p 8000:8000 \
  -e GOOGLE_APPLICATION_CREDENTIALS_JSON='<your-json-credentials>' \
  -e COHERE_API_KEY='<your-cohere-key>' \
  prompt-detective
```

### Deployment Options

The service is configured for deployment on:
- Google Cloud Run
- AWS ECS/Fargate
- Azure Container Instances
- Any Kubernetes cluster

See `part2/deploy.sh` for platform-specific deployment scripts.

## 🔧 API Usage

### Analyze Endpoint

```bash
POST /analyze
```

Request body:
```json
{
  "output_text": "The text to analyze",
  "max_attempts": 5,
  "context": {
    "domain": "technical",
    "complexity": "high"
  }
}
```

Response:
```json
{
  "success": true,
  "result": {
    "best_hypothesis": {
      "prompt": "The reconstructed prompt",
      "confidence": 0.85,
      "reasoning": "Detailed explanation",
      "key_elements": ["element1", "element2"]
    },
    "alternative_hypotheses": [...],
    "confidence": "high",
    "enhanced_scoring": {
      "semantic_similarity": 0.92,
      "structural_match": 0.88,
      "confidence_interval": [0.82, 0.94]
    }
  }
}
```

## 🎯 System Design

### Agent Architecture

The Prompt Detective agent uses a sophisticated multi-tool approach:

1. **Pattern Analyzer**: Extracts linguistic patterns, constraints, and structural elements
2. **Hypothesis Generator**: Uses Gemini to generate multiple prompt candidates
3. **Validator**: Employs Cohere for semantic similarity and constraint validation
4. **Enhanced Validator**: Provides detailed 6-dimensional scoring

### Refinement Process

The agent performs up to 5 passes:
1. Initial pattern extraction and hypothesis generation
2. Validation and scoring of hypotheses
3. Refinement based on missing elements
4. Re-validation with enhanced scoring
5. Final optimization and confidence calibration

### Scoring Dimensions

1. **Semantic Similarity**: How well the meaning aligns
2. **Structural Match**: Format and organization similarity
3. **Constraint Satisfaction**: Specific requirements met
4. **Style Match**: Tone and writing style alignment
5. **Intent Preservation**: Core purpose maintained
6. **Complexity Penalty**: Favor simpler prompts

## 🔒 Security & Performance

- **Input Validation**: Strict Pydantic models prevent injection attacks
- **Rate Limiting**: Configurable limits on API usage
- **Timeout Protection**: Maximum 120s per analysis
- **Resource Limits**: Docker memory/CPU constraints
- **Error Handling**: Comprehensive error tracking and recovery

## 📊 Monitoring

The service includes:
- Health check endpoint
- Structured logging
- Performance metrics
- Error tracking
- Request/response logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📝 License

This project is part of a technical assessment and is provided as-is for evaluation purposes.

## 🚨 Troubleshooting

### Common Issues

1. **"Cohere ClientV2 not found"**: The code handles both ClientV2 and legacy Client. Ensure Cohere SDK is installed.

2. **"Google credentials not found"**: Set `GOOGLE_APPLICATION_CREDENTIALS` env var to point to your service account key file.

3. **"Port already in use"**: The start script automatically kills existing processes on port 8000.

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python part1/start_server.py
```

## 📞 Support

For issues or questions about this assessment, please refer to the implementation details in `part3/ANSWERS.md` or review the comprehensive test suite in `part1/tests/`.