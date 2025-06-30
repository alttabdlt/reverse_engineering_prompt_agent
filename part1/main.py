from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
import logging
from dotenv import load_dotenv
from datetime import datetime
import time
import subprocess
import signal

from models import AnalyzeRequest, AnalyzeResponse, ErrorResponse
from agents import PromptDetectiveAgent

# Load environment variables
load_dotenv()

# Force correct credentials path
correct_creds = "/Users/axel/Desktop/Coding-Projects/assessment/service-account-key.json"
if os.path.exists(correct_creds):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = correct_creds

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Reverse Prompt Engineering Detective",
    description="An intelligent agent that analyzes AI outputs to deduce the prompts that created them",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
agent = PromptDetectiveAgent()

# Request tracking
active_requests = {}

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Track request
    active_requests[request_id] = {
        "start_time": time.time(),
        "path": request.url.path
    }
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    # Clean up tracking
    if request_id in active_requests:
        del active_requests[request_id]
    
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    error_response = ErrorResponse(
        success=False,
        message="An unexpected error occurred",
        error_code="INTERNAL_ERROR",
        details={"error": str(exc)} if os.getenv("DEBUG") else None
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Prompt Detective API is running",
        "version": "1.0.0",
        "endpoints": ["/analyze", "/health", "/metrics"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "prompt-detective",
        "timestamp": datetime.utcnow().isoformat(),
        "active_requests": len(active_requests)
    }

@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint"""
    return {
        "active_requests": len(active_requests),
        "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
        "version": "1.0.0"
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_output(request: Request, analyze_request: AnalyzeRequest):
    """
    Analyze an AI-generated output to detect the prompt that created it.
    
    This endpoint uses a multi-pass approach:
    1. Pattern analysis to extract structural and linguistic features
    2. Hypothesis generation using Gemini
    3. Validation using semantic similarity
    """
    request_id = request.state.request_id
    logger.info(f"Starting analysis for request {request_id}")
    
    try:
        # Call the agent
        result = await agent.detect_prompt(
            output_text=analyze_request.output_text,
            max_attempts=analyze_request.max_attempts,
            context=analyze_request.context
        )
        
        # Create response
        response = AnalyzeResponse(
            success=True,
            message="Analysis completed successfully",
            request_id=request_id,
            result=result
        )
        
        logger.info(f"Analysis completed for request {request_id} with confidence: {result.confidence}")
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed for request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

def kill_port_8000():
    """Kill any process using port 8000 before startup."""
    try:
        # Get current process ID and parent process ID
        current_pid = os.getpid()
        parent_pid = os.getppid()
        
        # Find process using port 8000
        result = subprocess.run(
            ["lsof", "-ti:8000"], 
            capture_output=True, 
            text=True
        )
        
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid and int(pid) not in [current_pid, parent_pid]:  # Don't kill ourselves or parent
                    try:
                        # Check if it's a uvicorn process before killing
                        ps_result = subprocess.run(
                            ["ps", "-p", pid, "-o", "comm="],
                            capture_output=True,
                            text=True
                        )
                        process_name = ps_result.stdout.strip()
                        
                        # Only kill if it's actually blocking our port
                        if "python" in process_name.lower() or "uvicorn" in process_name.lower():
                            os.kill(int(pid), signal.SIGTERM)
                            logger.info(f"Killed process {pid} ({process_name}) using port 8000")
                            time.sleep(0.5)
                    except (ProcessLookupError, ValueError):
                        pass
            
    except Exception as e:
        logger.warning(f"Error checking port 8000: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize application state on startup"""
    # Kill any existing process on port 8000
    kill_port_8000()
    
    app.state.start_time = time.time()
    logger.info("Prompt Detective API started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Prompt Detective API shutting down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)