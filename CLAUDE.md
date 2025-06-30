# Project: Reverse Prompt Engineering Detective
# Technical Assessment

## Project Overview
An intelligent agent that performs **reverse prompt engineering** - analyzing AI-generated outputs to deduce the original prompts that created them. Uses pattern analysis, hypothesis generation, and validation with up to 5 refinement passes.

## Architecture Design

### Core Components
1. **Output Analyzer Tool**: Extracts patterns, constraints, and stylistic elements
2. **Prompt Hypothesis Generator**: Uses LLM to generate likely prompts
3. **Validation Engine**: Tests hypotheses against the original output

### Key Features
- Pattern extraction from text (tone, structure, constraints)
- Multi-hypothesis generation and ranking
- Validation through regeneration and similarity scoring
- Confidence metrics for predictions
- 5-pass progressive refinement system
- Side-by-side prompt comparison
- Enhanced 6-dimensional scoring

## Implementation Strategy

### Phase 1: FastAPI Service
- Pydantic models for request/response
- Agent orchestration with tool routing (5-pass system)
- System prompt for detective persona
- Automatic port conflict resolution

### Phase 2: Tools Development
- Pattern extraction algorithms
- LLM integration for hypothesis generation
- Similarity scoring mechanisms

### Phase 3: Evaluation Framework
- Test cases covering various prompt types
- Automated scoring system
- Edge case handling

## Test Case Categories
1. Simple instructions (e.g., "Write a haiku")
2. Complex constraints (tone + format + topic)
3. Role-based prompts (personas)
4. Technical prompts (code generation)
5. Creative prompts (storytelling)
6. Edge cases (ambiguous outputs)

## Success Metrics
- Prompt reconstruction accuracy
- Constraint detection rate
- False positive minimization
- Processing time efficiency

## Deployment Notes
- Docker containerization with multi-stage build
- Environment-based configuration
- Prometheus metrics integration
- Health check endpoints