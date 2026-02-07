#!/usr/bin/env python3
"""
Entry point for running llm_manager as a module.

Usage:
    python -m llm_manager
    python -m llm_manager --port 8000
    python -m llm_manager scan
    python -m llm_manager --help
"""

from llm_manager.cli import main

if __name__ == "__main__":
    main()
