#!/bin/bash

# Deployment script for Prompt Detective API
set -e

echo "🚀 Starting deployment of Prompt Detective API..."

# Check if required environment variables are set
check_env_vars() {
    local required_vars=("GOOGLE_CLOUD_PROJECT" "GOOGLE_APPLICATION_CREDENTIALS" "COHERE_API_KEY")
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            echo "❌ Error: $var is not set"
            echo "Please set all required environment variables before deployment"
            exit 1
        fi
    done
    
    echo "✅ All required environment variables are set"
}

# Build Docker image
build_image() {
    echo "🔨 Building Docker image..."
    docker build -t prompt-detective:latest -f part2/Dockerfile .
    echo "✅ Docker image built successfully"
}

# Run tests before deployment
run_tests() {
    echo "🧪 Running tests..."
    docker run --rm \
        -e GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT \
        -e COHERE_API_KEY=$COHERE_API_KEY \
        -v $GOOGLE_APPLICATION_CREDENTIALS:/app/credentials/service-account.json:ro \
        prompt-detective:latest \
        pytest tests/ -v
    echo "✅ All tests passed"
}

# Deploy with Docker Compose
deploy_compose() {
    echo "🐳 Deploying with Docker Compose..."
    docker-compose -f part2/docker-compose.yml down
    docker-compose -f part2/docker-compose.yml up -d
    echo "✅ Application deployed successfully"
}

# Wait for health check
wait_for_health() {
    echo "⏳ Waiting for application to be healthy..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ Application is healthy!"
            return 0
        fi
        
        attempt=$((attempt + 1))
        echo "Waiting... ($attempt/$max_attempts)"
        sleep 2
    done
    
    echo "❌ Application failed to become healthy"
    return 1
}

# Main deployment flow
main() {
    echo "Environment: ${DEPLOY_ENV:-development}"
    
    check_env_vars
    build_image
    
    if [ "${SKIP_TESTS}" != "true" ]; then
        run_tests
    fi
    
    deploy_compose
    wait_for_health
    
    echo "🎉 Deployment completed successfully!"
    echo "📍 API is available at: http://localhost:8000"
    echo "📊 API documentation: http://localhost:8000/docs"
}

# Run main function
main "$@"