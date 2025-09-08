#!/usr/bin/env python3
"""
LLM Router API Runner
=====================

Quick way to start the API server for manual testing and exploration.

Usage:
    python run_api.py              # Start with default settings
    python run_api.py --port 3000  # Start on different port
    python run_api.py --reload     # Enable auto-reload for development
"""

import argparse
import uvicorn
from llm_router.api.main import app

def main():
    parser = argparse.ArgumentParser(description="LLM Router API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])

    args = parser.parse_args()

    print("🚀 Starting LLM Router API...")
    print(f"📍 Server: http://{args.host}:{args.port}")
    print(f"📚 Docs: http://{args.host}:{args.port}/docs")
    print(f"🔄 Reload: {'enabled' if args.reload else 'disabled'}")
    print("=" * 50)

    uvicorn.run(
        "llm_router.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main()
