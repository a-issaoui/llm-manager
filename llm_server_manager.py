#!/usr/bin/env python3
"""
LLM Server Manager - Manage LLM Manager HTTP server as a background service

Reads configuration from llm_manager.yaml and provides commands to manage the server.

Usage:
    python llm_server_manager.py start [--port 8000] [--host 0.0.0.0] [--model model.gguf]
    python llm_server_manager.py stop
    python llm_server_manager.py restart [--port 8000] [--host 0.0.0.0]
    python llm_server_manager.py status
    python llm_server_manager.py logs [--lines 50]
    python llm_server_manager.py run-foreground [--port 8000]

Examples:
    # Start server in background with settings from llm_manager.yaml
    python llm_server_manager.py start

    # Start with specific model pre-loaded
    python llm_server_manager.py start --model Qwen2.5-3b-instruct-q4_k_m.gguf

    # Start on custom port (overrides config)
    python llm_server_manager.py start --port 8888 --host 0.0.0.0

    # Check if server is running
    python llm_server_manager.py status

    # View recent logs
    python llm_server_manager.py logs --lines 100

    # Stop the server
    python llm_server_manager.py stop

    # Restart with same settings
    python llm_server_manager.py restart

    # Run in foreground (for debugging)
    python llm_server_manager.py run-foreground
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests
import yaml

from llm_manager.server import LLMServer

# Files
PID_FILE = Path("/tmp/llm_manager_server.pid")
LOG_FILE = Path("/tmp/llm_manager_server.log")
CONFIG_FILE = Path(__file__).parent / "llm_manager.yaml"

# Setup logging for the manager itself
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from llm_manager.yaml."""
    default_config = {
        "server": {
            "host": "127.0.0.1",
            "port": 8888,
            "log_level": "info",
        },
        "models": {
            "dir": "./models",
            "use_subprocess": True,
        }
    }
    
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                config = yaml.safe_load(f)
                if config:
                    return config
        except Exception as e:
            logger.warning(f"Could not load {CONFIG_FILE}: {e}")
    
    return default_config


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent


def get_models_dir(config: dict) -> Path:
    """Get the models directory from config."""
    models_dir = config.get("models", {}).get("dir", "./models")
    if not Path(models_dir).is_absolute():
        return get_project_root() / models_dir
    return Path(models_dir)


def get_server_host(config: dict) -> str:
    """Get server host from config."""
    return config.get("server", {}).get("host", "127.0.0.1")


def get_server_port(config: dict) -> int:
    """Get server port from config."""
    return config.get("server", {}).get("port", 8888)


def get_log_level(config: dict) -> str:
    """Get log level from config."""
    # Map logging.level from config
    level = config.get("logging", {}).get("level", "INFO")
    return level.lower()


def is_server_running(host: str, port: int) -> bool:
    """Check if the server is running."""
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def get_server_pid() -> int | None:
    """Get the server PID from pid file."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            # Check if process actually exists
            if os.path.exists(f"/proc/{pid}"):
                return pid
            else:
                # Stale pid file
                PID_FILE.unlink()
        except (ValueError, OSError):
            PID_FILE.unlink(missing_ok=True)
    return None


def save_server_pid(pid: int):
    """Save server PID to file."""
    PID_FILE.write_text(str(pid))


def remove_pid_file():
    """Remove the pid file."""
    PID_FILE.unlink(missing_ok=True)


def start_server_background(
    config: dict,
    host: str | None = None,
    port: int | None = None,
    model: str | None = None,
) -> bool:
    """Start the server by spawning a new detached process."""
    
    # Use config values, allow CLI overrides
    host = host or get_server_host(config)
    port = port or get_server_port(config)
    log_level = get_log_level(config)
    models_dir = get_models_dir(config)
    
    # Check if already running
    if is_server_running(host, port):
        logger.error(f"Server is already running on http://{host}:{port}")
        return False
    
    # Check for stale pid file
    existing_pid = get_server_pid()
    if existing_pid:
        logger.warning(f"Found stale PID file (process {existing_pid} not running)")
        remove_pid_file()
    
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return False
    
    logger.info(f"Starting LLM Manager server...")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Models: {models_dir}")
    logger.info(f"  Config: {CONFIG_FILE}")
    if model:
        logger.info(f"  Preload model: {model}")
    
    # Create a wrapper script that will run the server
    wrapper_script = f"""
import sys
sys.path.insert(0, '{get_project_root()}')

from llm_manager.server import LLMServer

server = LLMServer(
    models_dir='{models_dir}',
    host='{host}',
    port={port},
    log_level='{log_level}'
)

# Save PID
with open('{PID_FILE}', 'w') as f:
    f.write(str(__import__('os').getpid()))

try:
    server.start(default_model={repr(model)})
except KeyboardInterrupt:
    server.stop()
"""
    
    wrapper_file = Path("/tmp/llm_server_wrapper.py")
    wrapper_file.write_text(wrapper_script)
    
    try:
        # Start process in background, detached
        process = subprocess.Popen(
            [sys.executable, str(wrapper_file)],
            stdout=open(LOG_FILE, "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Detach from parent
        )
        
        # Wait a moment for PID to be written
        time.sleep(1)
        
        # Wait for server to be ready
        logger.info("Waiting for server to be ready...")
        start_time = time.time()
        while time.time() - start_time < 60:
            if is_server_running(host, port):
                logger.info(f"✅ Server is running!")
                logger.info(f"   URL: http://{host}:{port}")
                logger.info(f"   Health: http://{host}:{port}/health")
                logger.info(f"   Docs: http://{host}:{port}/docs")
                logger.info(f"   Log: {LOG_FILE}")
                return True
            time.sleep(0.5)
        
        # Timeout - server didn't start
        logger.error("Server failed to start within 60 seconds")
        logger.error(f"Check logs: {LOG_FILE}")
        
        # Try to kill the process
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return False


def stop_server(host: str, port: int) -> bool:
    """Stop the running server."""
    
    pid = get_server_pid()
    
    if not pid and not is_server_running(host, port):
        logger.info("Server is not running")
        remove_pid_file()
        return True
    
    logger.info(f"Stopping LLM Manager server...")
    
    # Try graceful shutdown via API first
    if is_server_running(host, port):
        try:
            requests.get(f"http://{host}:{port}/health", timeout=5)
            logger.info("Server confirmed running, stopping...")
        except:
            pass
    
    # Kill the process
    if pid:
        try:
            # Try graceful termination first
            os.kill(pid, signal.SIGTERM)
            
            # Wait for process to stop
            for _ in range(10):
                if not is_server_running(host, port):
                    logger.info("✅ Server stopped")
                    remove_pid_file()
                    return True
                time.sleep(0.5)
            
            # Force kill if still running
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
            
            if not is_server_running(host, port):
                logger.info("✅ Server force-stopped")
                remove_pid_file()
                return True
            
        except ProcessLookupError:
            logger.info("✅ Server process not found (already stopped)")
            remove_pid_file()
            return True
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
            return False
    
    # Fallback: try to find and kill any llm_manager processes
    logger.warning("Attempting to find and stop server process...")
    try:
        subprocess.run(["pkill", "-f", "llm_server_wrapper.py"], check=False, capture_output=True)
        time.sleep(2)
        if not is_server_running(host, port):
            logger.info("✅ Server stopped")
            remove_pid_file()
            return True
    except Exception as e:
        logger.error(f"Error killing server: {e}")
    
    return False


def restart_server(
    config: dict,
    host: str | None = None,
    port: int | None = None,
    model: str | None = None
) -> bool:
    """Restart the server."""
    logger.info("Restarting LLM Manager server...")
    
    host = host or get_server_host(config)
    port = port or get_server_port(config)
    
    # Stop if running
    stop_server(host, port)
    time.sleep(1)
    
    # Start again
    return start_server_background(config, host, port, model)


def show_status(config: dict, host: str | None = None, port: int | None = None):
    """Show server status."""
    host = host or get_server_host(config)
    port = port or get_server_port(config)
    
    print("\n" + "="*60)
    print("LLM Manager Server Status")
    print("="*60)
    
    pid = get_server_pid()
    running = is_server_running(host, port)
    
    if running:
        print(f"✅ Server is RUNNING")
        print(f"   URL: http://{host}:{port}")
        
        try:
            response = requests.get(f"http://{host}:{port}/health", timeout=5)
            health = response.json()
            print(f"   Status: {health.get('status', 'unknown')}")
            print(f"   Model loaded: {health.get('model_loaded', False)}")
            print(f"   Current model: {health.get('current_model', 'none')}")
        except Exception as e:
            print(f"   Warning: Could not fetch health: {e}")
        
        if pid:
            print(f"   PID: {pid}")
        print(f"   Log file: {LOG_FILE}")
        print(f"   Config file: {CONFIG_FILE}")
        
        # Show recent log lines
        if LOG_FILE.exists():
            try:
                lines = LOG_FILE.read_text().splitlines()
                if lines:
                    print(f"\n   Recent log entries:")
                    for line in lines[-5:]:
                        print(f"     {line}")
            except:
                pass
        
    else:
        print(f"❌ Server is NOT RUNNING")
        print(f"   Config: {CONFIG_FILE}")
        print(f"   Expected: http://{host}:{port}")
        if pid:
            print(f"   Stale PID file found: {pid}")
    
    print("="*60 + "\n")


def show_logs(lines: int = 50):
    """Show server logs."""
    if not LOG_FILE.exists():
        print(f"No log file found: {LOG_FILE}")
        return
    
    try:
        log_content = LOG_FILE.read_text()
        log_lines = log_content.splitlines()
        
        print(f"\n{'='*80}")
        print(f"LLM Manager Server Logs (last {lines} lines)")
        print(f"{'='*80}\n")
        
        for line in log_lines[-lines:]:
            print(line)
        
        print(f"\n{'='*80}")
        print(f"Log file: {LOG_FILE}")
        print(f"Total lines: {len(log_lines)}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"Error reading logs: {e}")


def run_foreground(
    config: dict,
    host: str | None = None,
    port: int | None = None,
    model: str | None = None
):
    """Run server in foreground (for debugging)."""
    host = host or get_server_host(config)
    port = port or get_server_port(config)
    models_dir = get_models_dir(config)
    log_level = get_log_level(config)
    
    print("\n" + "="*60)
    print("Starting LLM Manager Server (foreground mode)")
    print("="*60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Models: {models_dir}")
    print(f"Config: {CONFIG_FILE}")
    print(f"Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    server = LLMServer(
        models_dir=str(models_dir),
        host=host,
        port=port,
        log_level=log_level,
    )
    
    try:
        server.start(default_model=model)
    except KeyboardInterrupt:
        print("\n\nStopping server...")
        server.stop()
        print("Server stopped.")


def main():
    # Load configuration
    config = load_config()
    default_host = get_server_host(config)
    default_port = get_server_port(config)
    
    parser = argparse.ArgumentParser(
        description="LLM Manager Server - Service Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Configuration file: {CONFIG_FILE}

Current settings from config:
  Host: {default_host}
  Port: {default_port}
  Models: {get_models_dir(config)}

Examples:
  %(prog)s start                           # Start with config settings
  %(prog)s start --port 8888               # Override port
  %(prog)s start --model model.gguf        # Start with model pre-loaded
  %(prog)s status                          # Check server status
  %(prog)s logs --lines 100                # Show last 100 log lines
  %(prog)s stop                            # Stop the server
  %(prog)s restart                         # Restart the server
  %(prog)s run-foreground                  # Run in foreground (debug)
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start server in background")
    start_parser.add_argument("--host", default=None, help=f"Server host (default: {default_host})")
    start_parser.add_argument("--port", type=int, default=None, help=f"Server port (default: {default_port})")
    start_parser.add_argument("--model", help="Pre-load specific model")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop server")
    stop_parser.add_argument("--host", default=None, help=f"Server host (default: {default_host})")
    stop_parser.add_argument("--port", type=int, default=None, help=f"Server port (default: {default_port})")
    
    # Restart command
    restart_parser = subparsers.add_parser("restart", help="Restart server")
    restart_parser.add_argument("--host", default=None, help=f"Server host (default: {default_host})")
    restart_parser.add_argument("--port", type=int, default=None, help=f"Server port (default: {default_port})")
    restart_parser.add_argument("--model", help="Pre-load specific model")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show server status")
    status_parser.add_argument("--host", default=None, help=f"Server host (default: {default_host})")
    status_parser.add_argument("--port", type=int, default=None, help=f"Server port (default: {default_port})")
    
    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Show server logs")
    logs_parser.add_argument("--lines", type=int, default=50, help="Number of lines to show")
    
    # Run foreground command
    fg_parser = subparsers.add_parser("run-foreground", help="Run server in foreground (debug)")
    fg_parser.add_argument("--host", default=None, help=f"Server host (default: {default_host})")
    fg_parser.add_argument("--port", type=int, default=None, help=f"Server port (default: {default_port})")
    fg_parser.add_argument("--model", help="Pre-load specific model")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "start":
        success = start_server_background(
            config=config,
            host=args.host,
            port=args.port,
            model=args.model,
        )
        return 0 if success else 1
    
    elif args.command == "stop":
        host = args.host or get_server_host(config)
        port = args.port or get_server_port(config)
        success = stop_server(host, port)
        return 0 if success else 1
    
    elif args.command == "restart":
        success = restart_server(config, args.host, args.port, args.model)
        return 0 if success else 1
    
    elif args.command == "status":
        show_status(config, args.host, args.port)
        return 0
    
    elif args.command == "logs":
        show_logs(args.lines)
        return 0
    
    elif args.command == "run-foreground":
        run_foreground(config, args.host, args.port, args.model)
        return 0
    
    return 0


if __name__ == "__main__":
    exit(main())
