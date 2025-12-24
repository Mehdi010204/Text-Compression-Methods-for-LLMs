import time
import pandas as pd
from .metrics import calculate_metrics


def run_benchmark(
    methods: dict,
    text: str,
    save_csv: str = "results/compression_benchmark.csv",
    return_texts: bool = False
):
    """
    Run all compression methods and compute metrics.
    
    Args:
        methods (dict): name → compression function
        text (str): input text
        save_csv (str): where to save the benchmark table
        return_texts (bool): if True, also return compressed texts

    Returns:
        df (DataFrame)
        compressed_outputs (dict) if return_texts=True
    """

    rows = []
    compressed_outputs = {}  # store compressed text by method

    for method_name, compress_func in methods.items():
        print(f"\n=== Running method: {method_name} ===")

        start = time.time()
        try:
            compressed_text, model_size_mb = compress_func(text)
        except Exception as e:
            print(f"ERROR in {method_name}: {e}")
            continue
        elapsed = time.time() - start

        # Save compressed text if asked
        if return_texts:
            compressed_outputs[method_name] = compressed_text

        # Compute metrics
        metrics = calculate_metrics(text, compressed_text, model_size_mb, elapsed)
        metrics["Method"] = method_name

        rows.append(metrics)

    # Build dataframe
    df = pd.DataFrame(rows)
    df = df[[
        "Method",
        "Tokens Before", "Tokens After", "Compression Ratio",
        "Semantic Similarity", "Compression Time (s)",
        "Model Size (MB)", "Total Cost"
    ]]

    # Save CSV
    df.to_csv(save_csv, index=False)
    print(f"Saved benchmark to {save_csv}")

    # Return depending on mode
    if return_texts:
        return df, compressed_outputs

    return df
