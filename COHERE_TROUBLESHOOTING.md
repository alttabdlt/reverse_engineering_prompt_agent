# Cohere API Troubleshooting Guide

## Common Issues and Solutions

### 1. Timeout Errors
**Error**: `Read timed out. (read timeout=120)`

**Solution**: The default timeout has been reduced from 120s to 30s to prevent long waits. If you still experience timeouts:
- Check your internet connection
- Verify the Cohere API is accessible
- Consider using shorter prompts for validation

### 2. Invalid Model Parameter
**Error**: `unknown field: parameter 'model' is not a valid field`

**Solution**: This has been fixed in two ways:
- The legacy Cohere Client's `generate()` method doesn't need the model parameter
- The legacy Cohere Client's `embed()` method also doesn't accept model parameter
- Fixed instance checking to properly detect Client vs ClientV2 types

### 3. API Key Issues
**Error**: `Invalid API key`

**Solution**:
1. Verify your API key is correct in `.env`:
   ```
   COHERE_API_KEY=your-actual-key-here
   ```
2. Check key has sufficient credits at [Cohere Dashboard](https://dashboard.cohere.com)
3. Regenerate key if needed

### 4. Rate Limiting
**Error**: `Rate limit exceeded`

**Solution**:
- Add delays between requests
- Upgrade your Cohere plan
- Use mock mode for testing without API calls

## Cohere API Version Compatibility

This project supports both:
- **Cohere v5.x** (Current) - Using legacy Client
- **Future ClientV2** - Ready when available

### Current Implementation Details

For Cohere v5.15.0:
```python
# Generate text (no model parameter)
response = client.generate(
    prompt="Your prompt",
    max_tokens=300,
    temperature=0.3,
    truncate='END'
)

# Get embeddings (no model parameter) 
response = client.embed(
    texts=["text1", "text2"]
    # No model, input_type, or embedding_types parameters
)
```

**Important**: The code now properly detects Client vs ClientV2 using instance checking:
```python
if hasattr(self.client, '__class__') and self.client.__class__.__name__ == 'ClientV2':
    # Use ClientV2 API with model parameter
else:
    # Use legacy Client API without model parameter
```

### Timeout Configuration

The validators now use a 30-second timeout instead of the default 120 seconds:
```python
client = cohere.Client(api_key, timeout=30)
```

This prevents long waits and provides faster feedback on connection issues.

## Testing Without Cohere

To test without using Cohere API credits:
1. Don't set `COHERE_API_KEY` in `.env`
2. The system will use mock validation mode
3. You'll see "using mock mode" in the logs

## Monitoring API Usage

Check your usage at: https://dashboard.cohere.com

The project uses:
- `generate()` endpoint for text generation
- `embed()` endpoint for semantic similarity

Each validation pass uses approximately:
- 1-2 generate calls
- 1-2 embed calls

## Debug Mode

Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
python interactive_demo.py
```

This will show:
- Exact API calls being made
- Response times
- Any API errors in detail