#!/usr/bin/env python3
"""Start the FastAPI server, killing any existing process on port 8000 first."""

import os
import sys
import subprocess
import time
import signal

def kill_port_8000():
    """Kill any process using port 8000."""
    try:
        # Find process using port 8000
        result = subprocess.run(
            ["lsof", "-ti:8000"], 
            capture_output=True, 
            text=True
        )
        
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"✅ Killed process {pid} using port 8000")
                except ProcessLookupError:
                    pass
            
            # Wait a moment for port to be released
            time.sleep(1)
        else:
            print("✅ Port 8000 is free")
            
    except Exception as e:
        print(f"⚠️  Error checking port: {e}")

def start_server():
    """Start the uvicorn server."""
    print("🚀 Starting Prompt Detective API...")
    
    # Kill any existing process on port 8000
    kill_port_8000()
    
    # Start uvicorn
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--reload", 
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    start_server()