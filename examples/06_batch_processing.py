#!/usr/bin/env python3
"""
Example 06: Batch Processing

Demonstrates processing multiple prompts efficiently using the same model.
Uses Reason-With-Choice model for classification tasks.
"""

import logging
import time
from pathlib import Path

from llm_manager import LLMManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def classify_sentiment(manager: LLMManager, text: str) -> str:
    """Classify sentiment of a text."""
    messages = [{
        "role": "user",
        "content": f"""Classify the sentiment of the following text as POSITIVE, NEGATIVE, or NEUTRAL.

Text: "{text}"

Sentiment:"""
    }]
    
    response = manager.generate(
        messages=messages,
        max_tokens=32,
        temperature=0.1,  # Low temperature for classification
        stop=["\n", "."]
    )
    
    message = response.get("choices", [{}])[0].get("message", {})
    return message.get("content", "UNKNOWN").strip()


def summarize_text(manager: LLMManager, text: str) -> str:
    """Summarize a text."""
    messages = [{
        "role": "user",
        "content": f"""Summarize the following text in one sentence:

{text}

Summary:"""
    }]
    
    response = manager.generate(
        messages=messages,
        max_tokens=64,
        temperature=0.3,
        stop=["\n\n"]
    )
    
    message = response.get("choices", [{}])[0].get("message", {})
    return message.get("content", "").strip()


def main():
    """Run batch processing example."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    # Use Reason-With-Choice model
    model_name = "Reason-With-Choice-3B.Q4_K_M.gguf"
    
    logger.info("Initializing batch processing example")
    
    manager = LLMManager(
        models_dir=str(models_dir),
        use_subprocess=False,
        verbose=True
    )
    
    logger.info(f"Loading model: {model_name}")
    
    success = manager.load_model(model_name, config={
        "n_ctx": 4096,
        "n_gpu_layers": 0,
    })
    
    if not success:
        logger.error("Failed to load model!")
        return 1
    
    # Sample texts for batch processing
    texts = [
        "I absolutely love this product! Best purchase I've ever made.",
        "The service was terrible and the staff was rude.",
        "It's an okay product, nothing special but does the job.",
        "Amazing quality and fast shipping. Highly recommended!",
        "Very disappointed with the quality. It broke after one day.",
    ]
    
    print("\n" + "="*70)
    print("BATCH SENTIMENT CLASSIFICATION")
    print("="*70)
    
    start_time = time.time()
    
    for i, text in enumerate(texts, 1):
        sentiment = classify_sentiment(manager, text)
        print(f"\n{i}. Text: {text[:50]}...")
        print(f"   Sentiment: {sentiment}")
    
    classification_time = time.time() - start_time
    print(f"\nProcessed {len(texts)} texts in {classification_time:.2f}s")
    
    # Batch summarization
    articles = [
        "Artificial intelligence is transforming the way we work and live. "
        "From self-driving cars to medical diagnosis, AI is making significant "
        "impacts across all industries. However, concerns about job displacement "
        "and privacy remain important topics of discussion.",
        
        "Climate change poses one of the greatest challenges to humanity. "
        "Rising temperatures, extreme weather events, and sea level rise "
        "threaten ecosystems and communities worldwide. Urgent action is needed "
        "to reduce greenhouse gas emissions and transition to renewable energy.",
    ]
    
    print("\n" + "="*70)
    print("BATCH SUMMARIZATION")
    print("="*70)
    
    start_time = time.time()
    
    for i, article in enumerate(articles, 1):
        summary = summarize_text(manager, article)
        print(f"\n{i}. Original ({len(article)} chars):")
        print(f"   {article[:80]}...")
        print(f"\n   Summary ({len(summary)} chars):")
        print(f"   {summary}")
    
    summarization_time = time.time() - start_time
    print(f"\nProcessed {len(articles)} articles in {summarization_time:.2f}s")
    
    print("="*70)
    
    # Cleanup
    manager.unload_model()
    logger.info("Done!")
    
    return 0


if __name__ == "__main__":
    exit(main())
