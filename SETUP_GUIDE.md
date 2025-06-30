# 🔐 Credentials Setup Guide

## Quick Fix for "service-account-key.json not found"

### Option 1: Use the Setup Script (Recommended)
```bash
python setup_credentials.py
```
This interactive script will guide you through the setup process.

### Option 2: Manual Setup

#### Method A: Update the Path
If you have a service account JSON file somewhere else:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/actual/service-account.json
```

#### Method B: Use JSON Content Directly
```bash
export GOOGLE_APPLICATION_CREDENTIALS_JSON='<paste-your-json-here>'
```

Or more safely:
```bash
export GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat /path/to/your/service-account.json)"
```

#### Method C: Use .env File
1. Copy the example:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and update:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account.json
   GOOGLE_CLOUD_PROJECT=your-project-id
   ```

### Getting a Service Account Key

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Select your project
3. Navigate to "IAM & Admin" → "Service Accounts"
4. Create a new service account or select existing
5. Click "Keys" → "Add Key" → "Create new key" → JSON
6. Download the JSON file

### Required Permissions
The service account needs:
- `Vertex AI User` role
- Access to Gemini models in your selected region

### Verify Setup
After setting credentials, test with:
```bash
python -c "
import os
from google.oauth2 import service_account
import vertexai
from vertexai.preview.generative_models import GenerativeModel

# Test credentials
creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if creds_path and os.path.exists(creds_path):
    print(f'✓ Found credentials at: {creds_path}')
else:
    print('✗ Credentials file not found')

# Test Vertex AI
try:
    vertexai.init(project=os.getenv('GOOGLE_CLOUD_PROJECT', 'test'))
    print('✓ Vertex AI initialized')
except Exception as e:
    print(f'✗ Vertex AI error: {e}')
"
```

### Common Issues

1. **"Application Default Credentials not found"**
   - Run: `gcloud auth application-default login`
   - Or set GOOGLE_APPLICATION_CREDENTIALS

2. **"Permission denied"**
   - Check service account has Vertex AI User role
   - Verify billing is enabled

3. **"Invalid JSON"**
   - Ensure the file is a valid service account key
   - Check for proper JSON formatting

### Environment Variables Summary
```bash
# Required
export GOOGLE_CLOUD_PROJECT="your-project-id"

# One of these:
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
# OR
export GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":"service_account",...}'

# Optional
export VERTEX_AI_LOCATION="us-central1"  # default
```

After setting up, run:
```bash
python interactive_demo.py
```