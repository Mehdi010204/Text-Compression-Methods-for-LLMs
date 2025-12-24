from transformers import AutoTokenizer
from sentence_transformers import util
from compression.utils import embedder
from .chunking import encode_long_text
import time

# Use GPT-2 tokenizer for token count = consistent across all tests
tokenizer = AutoTokenizer.from_pretrained("gpt2")


def calculate_metrics(original: str, compressed: str, model_size_mb: int, compression_time: float):
    """
    Compute evaluation metrics for text compression methods.
    """

    # Token counts
    tokens_before = len(tokenizer.encode(original))
    tokens_after = len(tokenizer.encode(compressed))
    compression_ratio = tokens_after / tokens_before

    # Semantic similarity (chunk-safe)
    emb_orig = encode_long_text(original)
    emb_comp = encode_long_text(compressed)
    semantic_similarity = float(util.cos_sim(emb_orig, emb_comp))

    # Total cost (tunable weights)
    alpha, beta, gamma, delta = 1.0, 0.01, 0.001, 1.0
    total_cost = (
        alpha * tokens_after +
        beta * compression_time +
        gamma * model_size_mb +
        delta * (1 - semantic_similarity)
    )

    return {
        "Tokens Before": tokens_before,
        "Tokens After": tokens_after,
        "Compression Ratio": compression_ratio,
        "Semantic Similarity": semantic_similarity,
        "Compression Time (s)": compression_time,
        "Model Size (MB)": model_size_mb,
        "Total Cost": total_cost,
    }
