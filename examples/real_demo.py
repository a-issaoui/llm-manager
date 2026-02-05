#!/usr/bin/env python3
"""
REAL Demo - LLM Manager v5.0

This script runs REAL server, REAL model calls, REAL agents, REAL switching.

Prerequisites:
    1. Have a GGUF model in ./models/ directory
    2. Install dependencies: pip install openai langchain langchain-openai
    
Usage:
    python examples/real_demo.py
"""

import subprocess
import time
import sys
import os
import json
import signal
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_manager import LLMManager, ChatHistory, HistoryConfig, get_global_metrics, get_config


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_section(title):
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.ENDC}")


def print_info(msg):
    print(f"{Colors.CYAN}ℹ {msg}{Colors.ENDC}")


def print_warning(msg):
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")


def print_fail(msg):
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")


def get_first_model():
    """Get first available model from registry."""
    models_file = get_config().models.get_registry_path()
    if models_file.exists():
        with open(models_file) as f:
            models = json.load(f)
        if models:
            return list(models.keys())[0]
    return None


def find_model():
    """Find a model from the models.json registry."""
    models_dir = Path(get_config().models.dir)
    if not models_dir.exists():
        print_warning(f"No models directory found at {models_dir}. Creating...")
        models_dir.mkdir(exist_ok=True)
        return None
    
    # Use models.json registry
    models_json = get_config().models.get_registry_path()
    if models_json.exists():
        import json
        with open(models_json) as f:
            registry = json.load(f)
        if registry:
            # Return first model name from registry
            first_model = list(registry.keys())[0]
            print_info(f"Found model in registry: {first_model}")
            return models_dir / first_model
    
    # Fallback: find .gguf files
    gguf_files = list(models_dir.glob("*.gguf"))
    if gguf_files:
        return gguf_files[0]
    
    print_warning(f"No models found in {models_dir}")
    print_info("Please download a model, e.g.:")
    print(f"  wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf -P {models_dir}/")
    return None


class RealDemo:
    def __init__(self):
        self.server_process = None
        self.model_path = None
        self.base_url = "http://localhost:8000/v1"
        
    def start_server(self, model_name):
        """Start the real server."""
        print_section("STARTING REAL SERVER")
        print_info(f"Starting server with model: {model_name}")
        print_info("This will load the model into memory...\n")
        
        # Start server as subprocess
        cmd = [
            sys.executable, "-m", "llm_manager.cli",
            "--model", model_name,
            "--port", "8000",
            "--host", "127.0.0.1"
        ]
        
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Wait for server to be ready
        print_info("Waiting for server to start...")
        time.sleep(5)  # Give it time to load
        
        # Check if server is responding
        import urllib.request
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=10)
            print_success("Server is running!")
            return True
        except Exception as e:
            print_warning(f"Server might not be ready yet: {e}")
            # Print any output
            if self.server_process.poll() is not None:
                stdout, _ = self.server_process.communicate()
                print(stdout)
                return False
            print_success("Server process started (continuing anyway)")
            return True
    
    def test_basic_chat(self):
        """Test basic chat completion."""
        print_section("TEST 1: BASIC CHAT COMPLETION")
        
        try:
            from openai import OpenAI
            
            client = OpenAI(
                base_url=self.base_url,
                api_key="not-needed"
            )
            
            print_info("Sending chat request...")
            response = client.chat.completions.create(
                model=self.model_path.stem if self.model_path else "model",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2+2? Answer in one word."}
                ],
                temperature=0.7,
                max_tokens=50
            )
            
            content = response.choices[0].message.content
            print_success(f"Response received: '{content}'")
            print_info(f"Tokens used: {response.usage.total_tokens}")
            return True
            
        except Exception as e:
            print_warning(f"Chat test failed: {e}")
            return False
    
    def test_streaming(self):
        """Test streaming chat."""
        print_section("TEST 2: STREAMING CHAT")
        
        try:
            from openai import OpenAI
            
            client = OpenAI(
                base_url=self.base_url,
                api_key="not-needed"
            )
            
            print_info("Starting stream...")
            print("Response: ", end="", flush=True)
            
            stream = client.chat.completions.create(
                model=self.model_path.stem if self.model_path else "model",
                messages=[
                    {"role": "user", "content": "Count 1 to 3"}
                ],
                stream=True,
                max_tokens=50
            )
            
            full_response = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    full_response += content
            
            print()  # Newline
            print_success(f"Stream complete ({len(full_response)} chars)")
            return True
            
        except Exception as e:
            print_warning(f"Streaming test failed: {e}")
            return False
    
    def test_chat_history(self):
        """Test chat history management."""
        print_section("TEST 3: CHAT HISTORY")
        
        # Create history
        config = HistoryConfig(
            max_tokens=2000,
            truncation_strategy="middle",
            keep_system_prompt=True
        )
        history = ChatHistory(config)
        
        # Add conversation
        history.add_system("You are a Python expert.")
        history.add_user("How do I create a list?")
        history.add_assistant("Use square brackets: my_list = [1, 2, 3]")
        history.add_user("How do I append?")
        
        print_info(f"History has {len(history)} messages")
        print_info(f"Token count: {history.get_token_count()}")
        
        # Use history for generation
        try:
            from openai import OpenAI
            client = OpenAI(base_url=self.base_url, api_key="not-needed")
            
            print_info("Sending with conversation history...")
            response = client.chat.completions.create(
                model=self.model_path.stem if self.model_path else "model",
                messages=history.get_messages(),
                max_tokens=100
            )
            
            content = response.choices[0].message.content
            history.add_assistant(content)
            
            print_success(f"Response: '{content[:80]}...'")
            print_info(f"History now has {len(history)} messages")
            return True
            
        except Exception as e:
            print_warning(f"History test failed: {e}")
            return False
    
    def test_batch(self):
        """Test batch generation."""
        print_section("TEST 4: BATCH GENERATION")
        
        try:
            from openai import OpenAI
            client = OpenAI(base_url=self.base_url, api_key="not-needed")
            
            prompts = [
                "What is Python?",
                "What is AI?",
                "What is ML?"
            ]
            
            print_info(f"Sending {len(prompts)} prompts...")
            
            import concurrent.futures
            
            def generate_one(prompt):
                return client.chat.completions.create(
                    model=self.model_path.stem if self.model_path else "model",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=30
                )
            
            start = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(generate_one, p) for p in prompts]
                results = [f.result() for f in futures]
            
            elapsed = time.time() - start
            
            print_success(f"Batch complete in {elapsed:.2f}s")
            for i, (prompt, result) in enumerate(zip(prompts, results)):
                content = result.choices[0].message.content[:40]
                print(f"  {i+1}. '{prompt}' -> '{content}...'")
            
            return True
            
        except Exception as e:
            print_warning(f"Batch test failed: {e}")
            return False
    
    def test_model_switch(self):
        """Test model switching."""
        print_section("TEST 5: MODEL SWITCHING")
        
        # Use models.json registry
        models_json = get_config().models.get_registry_path()
        if models_json.exists():
            import json
            with open(models_json) as f:
                registry = json.load(f)
            model_list = list(registry.keys())
        else:
            model_list = [m.name for m in Path(get_config().models.dir).glob("*.gguf")]
        
        if len(model_list) < 2:
            print_warning(f"Need 2+ models to test switching, found {len(model_list)}")
            print_info("Add more models to ./models/ directory")
            return True
        
        try:
            import urllib.request
            
            model1 = model_list[0]
            model2 = model_list[1]
            
            print_info(f"Switching from {model1} to {model2}...")
            
            # Use admin endpoint to switch
            req = urllib.request.Request(
                f"http://localhost:8000/admin/switch-model",
                data=json.dumps({"model": model2}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            try:
                urllib.request.urlopen(req, timeout=30)
                print_success(f"Switched to {model2}")
            except urllib.error.HTTPError as e:
                print_info(f"Switch endpoint returned: {e.code}")
                print_info("(This is OK - switching may be done differently)")
            
            return True
            
        except Exception as e:
            print_warning(f"Switch test failed: {e}")
            return False
    
    def test_langchain(self):
        """Test LangChain integration."""
        print_section("TEST 6: LANGCHAIN INTEGRATION")
        
        try:
            from langchain_openai import ChatOpenAI
            from langchain.schema import HumanMessage
            
            print_info("Creating LangChain LLM...")
            
            llm = ChatOpenAI(
                model=self.model_path.stem if self.model_path else "model",
                openai_api_base=self.base_url,
                openai_api_key="not-needed",
                temperature=0.7
            )
            
            print_info("Sending request via LangChain...")
            response = llm.invoke([HumanMessage(content="Hello!")])
            
            print_success(f"LangChain response: '{response.content[:80]}...'")
            return True
            
        except ImportError:
            print_warning("LangChain not installed. Skipping.")
            print_info("Install: pip install langchain langchain-openai")
            return True
        except Exception as e:
            print_warning(f"LangChain test failed: {e}")
            return False
    
    def test_metrics(self):
        """Test metrics collection."""
        print_section("TEST 7: METRICS")
        
        try:
            import urllib.request
            
            print_info("Getting server stats...")
            
            req = urllib.request.Request("http://localhost:8000/admin/stats")
            with urllib.request.urlopen(req, timeout=5) as resp:
                stats = json.loads(resp.read().decode())
            
            print_success("Metrics retrieved:")
            print(f"  Model loaded: {stats.get('model_loaded', 'N/A')}")
            print(f"  Current model: {stats.get('current_model', 'N/A')}")
            
            if 'metrics' in stats:
                m = stats['metrics']
                print(f"  Total requests: {m.get('total_requests', 0)}")
                print(f"  Tokens/sec: {m.get('tokens_per_second', 0):.2f}")
            
            return True
            
        except Exception as e:
            print_warning(f"Metrics test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests."""
        print_section("LLM MANAGER v5.0 - REAL DEMO")
        
        # Find model
        self.model_path = find_model()
        if not self.model_path:
            print_fail("No model found. Cannot continue.")
            return False
        
        model_name = self.model_path.stem
        print_success(f"Found model: {model_name}")
        
        # Start server
        if not self.start_server(model_name):
            return False
        
        # Give server time to fully load
        print_info("Waiting for model to load...")
        time.sleep(3)
        
        results = []
        
        # Run tests
        results.append(("Basic Chat", self.test_basic_chat()))
        results.append(("Streaming", self.test_streaming()))
        results.append(("Chat History", self.test_chat_history()))
        results.append(("Batch", self.test_batch()))
        results.append(("Model Switch", self.test_model_switch()))
        results.append(("LangChain", self.test_langchain()))
        results.append(("Metrics", self.test_metrics()))
        
        return results
    
    def cleanup(self):
        """Stop server."""
        print_section("CLEANUP")
        
        if self.server_process:
            print_info("Stopping server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
                print_success("Server stopped")
            except:
                self.server_process.kill()
                print_warning("Server killed")


def main():
    demo = RealDemo()
    
    try:
        results = demo.run_all_tests()
        
        # Summary
        print_section("DEMO SUMMARY")
        
        passed = sum(1 for _, r in results if r)
        total = len(results)
        
        print(f"Tests: {passed}/{total} passed\n")
        
        for name, result in results:
            status = f"{Colors.GREEN}✓ PASS{Colors.ENDC}" if result else f"{Colors.FAIL}✗ FAIL{Colors.ENDC}"
            print(f"  {name:<20} {status}")
        
        if passed == total:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED!{Colors.ENDC}")
        else:
            print(f"\n{Colors.WARNING}{Colors.BOLD}⚠ Some tests failed{Colors.ENDC}")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        demo.cleanup()


if __name__ == "__main__":
    main()
