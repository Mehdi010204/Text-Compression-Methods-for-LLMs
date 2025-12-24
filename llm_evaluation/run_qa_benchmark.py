import os
import json

from qa_evaluator import evaluate_single_context
from semantic_similarity import semantic_similarity


def main():

    # ------------------------
    # LLM used for evaluation
    # ------------------------
    model = "llama3.2"

    # ------------------------
    # Load the full article
    # ------------------------
    article_path = "C:/Users/midou/OneDrive/Bureau/Text Compression for LLMs/quantitative_eval/data/sample_texts/article.txt"

    if not os.path.exists(article_path):
        raise FileNotFoundError(f"The full article file is missing: {article_path}")

    with open(article_path, "r", encoding="utf8") as f:
        full_context = f.read()

    # ------------------------
    # Define the question (same for all tests)
    # ------------------------
    question = "What are the main characteristics of traditional Japanese arts?"

    print("\n=== BASELINE (full context) ===\n")

    baseline_answer, baseline_latency = evaluate_single_context(
        model=model,
        question=question,
        context=full_context
    )

    print("Baseline Answer:")
    print(baseline_answer)
    print("Baseline Latency:", round(baseline_latency, 3), "s")

    # ------------------------
    # Load compressed contexts
    # ------------------------
    methods = ["semanticpruning", "summarization", "crosslingual"]
    contexts = {}

    compressed_dir = "C:/Users/midou/OneDrive/Bureau/Text Compression for LLMs/quantitative_eval/data/compressed"

    for m in methods:
        path = f"{compressed_dir}/{m}.txt"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Compressed context missing: {path}")

        with open(path, "r", encoding="utf8") as f:
            contexts[m] = f.read()

    # ------------------------
    # Evaluate all compression methods
    # ------------------------
    print("\n\n=== RUNNING LLM QA EVALUATION ===\n")

    results = []

    for m in methods:
        print(f"\n=== {m} ===")

        answer, latency = evaluate_single_context(
            model=model,
            question=question,
            context=contexts[m]
        )

        # Compute semantic drift (similarity)
        quality = semantic_similarity(baseline_answer, answer)

        print("\nCompressed Answer:")
        print(answer)
        print("Latency:", round(latency, 3), "s")
        print("Quality vs Baseline:", round(quality, 3))

        results.append({
            "method": m,
            "latency": latency,
            "quality": quality,
            "answer": answer
        })

    # ------------------------
    # Save everything to JSON
    # ------------------------
    out_path = "C:/Users/midou/OneDrive/Bureau/Text Compression for LLMs/llm_evaluation/results/llm_evaluation_results.json"

    os.makedirs("results", exist_ok=True)

    with open(out_path, "w", encoding="utf8") as f:
        json.dump({
            "baseline_answer": baseline_answer,
            "baseline_latency": baseline_latency,
            "results": results
        }, f, indent=4)

    print(f"\nSaved LLM evaluation results\n")


if __name__ == "__main__":
    main()
