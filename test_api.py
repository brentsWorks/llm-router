#!/usr/bin/env python3
"""
LLM Router API Manual Tester
============================

Manual testing script for the LLM Router API.
Use this to experiment with different prompts and see how the API responds.

Usage:
    python test_api.py                           # Test all scenarios
    python test_api.py --prompt "Write Python code"  # Test specific prompt
    python test_api.py --health                   # Just test health check
"""

import argparse
import json
import time
from typing import List, Dict, Any
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=30.0)

    def test_health(self) -> bool:
        """Test the health check endpoint."""
        try:
            start_time = time.time()
            response = self.client.get("/health")
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                console.print(f"✅ Health Check: {data['status']} ({response_time:.2f}s)")
                return True
            else:
                console.print(f"❌ Health Check: {response.status_code}")
                return False
        except Exception as e:
            console.print(f"❌ Health Check Error: {e}")
            return False

    def test_route(self, prompt: str) -> Dict[str, Any]:
        """Test the routing endpoint with a specific prompt."""
        try:
            payload = {"prompt": prompt}
            start_time = time.time()
            response = self.client.post("/route", json=payload)
            response_time = time.time() - start_time

            result = {
                "prompt": prompt,
                "status_code": response.status_code,
                "response_time": response_time,
                "data": None,
                "error": None
            }

            if response.status_code == 200:
                result["data"] = response.json()
            else:
                try:
                    result["error"] = response.json()
                except:
                    result["error"] = response.text

            return result
        except Exception as e:
            return {
                "prompt": prompt,
                "status_code": 0,
                "response_time": 0,
                "data": None,
                "error": str(e)
            }

    def display_result(self, result: Dict[str, Any]):
        """Display a test result in a nice format."""
        prompt = result["prompt"]
        status = result["status_code"]
        response_time = result["response_time"]

        if status == 200 and result["data"]:
            data = result["data"]
            model = data["selected_model"]
            classification = data["classification"]

            # Create a nice display
            table = Table(title=f"📝 Prompt: {prompt[:50]}...")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Status", f"✅ {status} ({response_time:.2f}s)")
            table.add_row("Selected Model", f"{model['provider']}/{model['model']}")
            table.add_row("Confidence", ".2%")
            table.add_row("Category", classification['category'])
            table.add_row("Classification Confidence", ".2%")

            console.print(table)

            if data.get("reasoning"):
                console.print(f"💭 Reasoning: {data['reasoning']}")

        else:
            error_msg = result.get("error", "Unknown error")
            console.print(f"❌ Error ({status}): {error_msg}")

    def run_test_suite(self, prompts: List[str]):
        """Run a comprehensive test suite."""
        console.print(Panel.fit("🧪 LLM Router API Test Suite", style="bold blue"))

        # Test health
        console.print("\n🏥 Testing Health Check...")
        health_ok = self.test_health()

        if not health_ok:
            console.print("❌ Health check failed - is the API server running?")
            return

        # Test routing
        console.print("\n🚀 Testing Routing Endpoints...")
        for prompt in prompts:
            console.print(f"\n🔍 Testing: {prompt[:40]}...")
            result = self.test_route(prompt)
            self.display_result(result)
            time.sleep(0.5)  # Small delay between requests

def main():
    parser = argparse.ArgumentParser(description="LLM Router API Tester")
    parser.add_argument("--prompt", help="Test a specific prompt")
    parser.add_argument("--health", action="store_true", help="Only test health check")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")

    args = parser.parse_args()

    tester = APITester(args.url)

    # Default test prompts
    test_prompts = [
        "Write a Python function to calculate fibonacci numbers",
        "Tell me a creative story about a magical forest",
        "What is the difference between machine learning and deep learning?",
        "Explain quantum computing in simple terms",
        "Help me debug this JavaScript code",
        "Write a poem about artificial intelligence",
        "How do I optimize my database queries?",
        "Create a recipe for chocolate chip cookies",
        "",  # Empty prompt test
        "A" * 1000,  # Very long prompt test
    ]

    if args.health:
        tester.test_health()
    elif args.prompt:
        result = tester.test_route(args.prompt)
        tester.display_result(result)
    else:
        tester.run_test_suite(test_prompts)

if __name__ == "__main__":
    main()
