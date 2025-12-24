import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmark.runner import run_benchmark
from compression.semantic_pruning import compress as semantic_pruning
from compression.summarization import compress as summarization
from compression.crosslingual import compress as crosslingual


def load_text(path: str) -> str:
    """Load a text file safely."""
    with open(path, "r", encoding="utf8") as f:
        return f.read()


def save_compressed_text(method_name: str, text: str):
    """Save compressed text into data/compressed/<method>.txt."""
    output_dir = "C:/Users/midou/OneDrive/Bureau/Text Compression for LLMs/quantitative_eval/data/compressed"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{method_name.lower()}.txt"  
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w", encoding="utf8") as f:
        f.write(text)

    print(f"[✓] Saved {method_name} compressed text → {output_path}")


def main():
    print('Using device:', os.environ.get('CUDA_VISIBLE_DEVICES', 'CPU'))

    # --- Define compression methods ---
    methods = {
        "SemanticPruning": semantic_pruning,
        "Summarization": summarization,
        "CrossLingual": crosslingual,
    }

    # --- Load input text ---
    input_path = "C:/Users/midou/OneDrive/Bureau/Text Compression for LLMs/quantitative_eval/data/sample_texts/article.txt"
    print(f"Loading input text from: {input_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"File not found: {input_path}\n"
            "Please add a text file in data/sample_texts/ named 'article.txt'"
        )

    text = load_text(input_path)

    # --- Run benchmark and get compressed outputs ---
    print("\nRunning benchmark...\n")
    results, compressed_texts = run_benchmark(
        methods,
        text,
        save_csv="C:/Users/midou/OneDrive/Bureau/Text Compression for LLMs/quantitative_eval/results/compression_benchmark.csv",
        return_texts=True
    )

    print("\n=== BENCHMARK RESULTS ===")
    print(results)

    print("\nSaving compressed outputs...\n")
    for method_name, comp_text in compressed_texts.items():
        save_compressed_text(method_name, comp_text)


if __name__ == "__main__":
    main()
