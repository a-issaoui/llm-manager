"""Setup script for llm-manager."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-manager",
    version="5.0.0",
    author="a-issaoui",
    author_email="",
    description="Production-grade LLM management system with OpenAI-compatible API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/a-issaoui/llm-manager",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "llama-cpp-python>=0.3.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
    ],
    extras_require={
        "agents": [
            "openai>=1.0.0",
            "pyautogen>=0.2.0",
            "langchain>=0.1.0",
            "langchain-openai>=0.0.5",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-timeout>=2.2.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-manager=llm_manager.cli:main",
        ],
    },
)
