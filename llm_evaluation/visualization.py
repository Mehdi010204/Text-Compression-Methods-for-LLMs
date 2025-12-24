import json
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------
# Load data
# ----------------------------------
with open("C:/Users/midou/OneDrive/Bureau/Text Compression for LLMs/llm_evaluation/results/llm_evaluation_results.json", "r", encoding="utf8") as f:
    data = json.load(f)

results = data["results"]

# ----------------------------------
# Load compression ratios from the compression benchmark
# ----------------------------------
import pandas as pd
df_comp = pd.read_csv("C:/Users/midou/OneDrive/Bureau/Text Compression for LLMs/quantitative_eval/results/compression_benchmark.csv")
compression_ratios = {
    row["Method"].lower(): row["Compression Ratio"]
    for _, row in df_comp.iterrows()
}

# ----------------------------------
# Prepare data for plotting
# ----------------------------------
methods = [r["method"] for r in results]

latencies = [r["latency"] for r in results]
qualities = [r["quality"] for r in results]
ratios = [1-compression_ratios[m.lower()] for m in methods]

x = np.arange(len(methods))
width = 0.25

# ----------------------------------
# Plot
# ----------------------------------
plt.figure(figsize=(12, 6))

plt.bar(x - width, ratios, width=width, label="Compression Ratio", alpha=0.8)
plt.bar(x, latencies, width=width, label="Latency (s)", alpha=0.8)
plt.bar(x + width, qualities, width=width, label="Semantic Similarity", alpha=0.8)

plt.xticks(x, methods, fontsize=12)
plt.ylabel("Metric Value", fontsize=12)
plt.title("Comparison of Compression Methods on LLM Evaluation", fontsize=14)
plt.legend(fontsize=12)

# plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
plt.savefig("C:/Users/midou/OneDrive/Bureau/Text Compression for LLMs/llm_evaluation/results/llm_evaluation_comparison.png", dpi=300)