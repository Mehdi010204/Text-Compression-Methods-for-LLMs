import time
from ollama_client import ask_ollama
from semantic_similarity import semantic_similarity


def evaluate_single_context(model: str, question: str, context: str):
    """Runs the LLM with a given context and returns answer + latency."""

    prompt = f"""
You must answer ONLY using the following context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    start = time.time()
    answer = ask_ollama(model, prompt)
    latency = time.time() - start
    return answer, latency
