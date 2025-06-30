#!/bin/bash

# Simple script to run the Prompt Detective API locally

echo "🔍 Starting Prompt Detective API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch venv/installed
fi

# Load environment variables if .env exists
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Change to part1 directory and run
cd part1
echo "Starting server on http://localhost:8000"
echo "API docs available at http://localhost:8000/docs"
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000