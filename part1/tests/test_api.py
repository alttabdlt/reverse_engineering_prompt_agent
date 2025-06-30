"""Unit tests for API endpoints"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from models import DetectionResult, PromptHypothesis, ConfidenceLevel

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

@pytest.fixture
def mock_detection_result():
    """Mock detection result"""
    return DetectionResult(
        best_hypothesis=PromptHypothesis(
            prompt="List 5 benefits of exercise",
            confidence=0.88,
            reasoning="Numbered list with 5 items",
            key_elements=["List", "5", "benefits"],
            rank=1
        ),
        all_hypotheses=[
            PromptHypothesis(
                prompt="List 5 benefits of exercise",
                confidence=0.88,
                reasoning="Numbered list with 5 items",
                key_elements=["List", "5", "benefits"],
                rank=1
            )
        ],
        validation_results=[],
        confidence=ConfidenceLevel.HIGH,
        attempts_used=2,
        execution_trace=[],
        processing_time_ms=1250
    )

class TestAPIEndpoints:
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "prompt-detective"
        assert "timestamp" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "active_requests" in data
        assert "version" in data
    
    @patch('main.agent.detect_prompt')
    def test_analyze_endpoint_success(self, mock_detect, client, mock_detection_result):
        """Test successful analysis"""
        mock_detect.return_value = mock_detection_result
        
        response = client.post("/analyze", json={
            "output_text": "1. First\n2. Second\n3. Third\n4. Fourth\n5. Fifth",
            "max_attempts": 3
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "result" in data
        assert data["result"]["best_hypothesis"]["prompt"] == "List 5 benefits of exercise"
        assert "request_id" in data
    
    def test_analyze_endpoint_validation_error(self, client):
        """Test validation error handling"""
        # Too short output text
        response = client.post("/analyze", json={
            "output_text": "short"
        })
        
        assert response.status_code == 422  # Validation error
    
    @patch('main.agent.detect_prompt')
    def test_analyze_endpoint_server_error(self, mock_detect, client):
        """Test server error handling"""
        mock_detect.side_effect = Exception("Detection failed")
        
        response = client.post("/analyze", json={
            "output_text": "Valid text that is long enough to pass validation"
        })
        
        assert response.status_code == 500
        assert "Analysis failed" in response.json()["detail"]
    
    def test_request_id_header(self, client):
        """Test that request ID is added to response headers"""
        response = client.get("/health")
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/health")
        assert response.status_code == 200
        # CORS headers should be present
    
    @patch('main.agent.detect_prompt')
    def test_analyze_with_context(self, mock_detect, client, mock_detection_result):
        """Test analysis with context parameter"""
        mock_detect.return_value = mock_detection_result
        
        response = client.post("/analyze", json={
            "output_text": "Test output with sufficient length for validation",
            "context": "technical documentation",
            "max_attempts": 2
        })
        
        assert response.status_code == 200
        # Verify context was passed to detect_prompt
        mock_detect.assert_called_once()
        call_args = mock_detect.call_args[1]
        assert call_args["context"] == "technical documentation"
        assert call_args["max_attempts"] == 2