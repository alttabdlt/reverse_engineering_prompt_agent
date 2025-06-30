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

## Known Issues & Solutions

### Over-fitting Problem (Critical)
The system currently suffers from severe over-fitting, reconstructing overly complex prompts when simple ones are more likely.

**Example**: 
- Original: "Explain how backpropagation works" (4 words)
- System reconstructs: 45-word prompt with bullet counts, word limits, etc.
- Accuracy: 0.6% ❌

**Root Cause**: System assumes every output characteristic was explicitly requested, not understanding that LLMs naturally add structure and formatting.

### Solution: Three-Pillar Approach

#### 1. Simplicity Scoring (Occam's Razor)
```python
# Update validator.py weights
weights = {
    'semantic': 0.4,     # Reduced from 0.5
    'simplicity': 0.3,   # NEW: Favor simple prompts
    'structural': 0.2,   # Reduced from 0.3
    'constraint': 0.1    # Reduced from 0.2
}
```

#### 2. Progressive Hints (for evaluation mode)
- Pass 1: No hints - try simplest approach first
- Pass 2: Length category (short/medium/long)
- Pass 3: Word count range + prompt type
- Pass 4: Starting word + masked preview
- Pass 5: Key terms + 40% revealed

#### 3. Better Hypothesis Generation
Update the system prompt in `hypothesis_generator.py`:
```
IMPORTANT PRINCIPLES:
1. Apply Occam's Razor - simpler prompts are more likely
2. LLMs naturally add formatting without being asked
3. Most user prompts are under 15 words
4. Start with simple patterns: "Explain X", "What is X?", "How does X work?"
```

### Implementation Priority
1. **Immediate Fix**: Update hypothesis generation prompt to prefer simplicity
2. **Quick Win**: Add `prompt_simplicity_scorer.py` to scoring pipeline
3. **Future**: Implement `progressive_hint_engine.py` for evaluation mode

### Key Insight
**LLMs are helpful by default** - they naturally add:
- Structure and formatting
- Comprehensive details
- Technical terminology
- Organization (bullets, sections)

The system must recognize these as emergent behaviors, not explicit requests.

### Target Metrics
- Simple prompts (<10 words): 80%+ accuracy
- Medium prompts (10-20 words): 60%+ accuracy
- Complex prompts (20+ words): 40%+ accuracy
- False complexity rate: <10%

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.