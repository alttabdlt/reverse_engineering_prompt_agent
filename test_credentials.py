#!/usr/bin/env python3
"""Quick test to verify credentials are working"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv(override=True)

# Check what we're getting
creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
print(f"GOOGLE_APPLICATION_CREDENTIALS: {creds_path}")

if creds_path and os.path.exists(creds_path):
    print("✓ Credentials file exists!")
    
    # Try to load it
    try:
        import json
        with open(creds_path) as f:
            data = json.load(f)
            print(f"✓ Valid JSON with type: {data.get('type', 'unknown')}")
            print(f"✓ Project ID in key: {data.get('project_id', 'not found')}")
    except Exception as e:
        print(f"✗ Error reading file: {e}")
else:
    print("✗ Credentials file not found!")

# Check other env vars
print(f"\nGOOGLE_CLOUD_PROJECT: {os.getenv('GOOGLE_CLOUD_PROJECT', 'not set')}")
print(f"VERTEX_AI_LOCATION: {os.getenv('VERTEX_AI_LOCATION', 'not set')}")
print(f"COHERE_API_KEY: {'set' if os.getenv('COHERE_API_KEY') else 'not set'}")