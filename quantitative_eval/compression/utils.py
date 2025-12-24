import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
from sentence_transformers import SentenceTransformer
import re

# Global embedder (GPU auto if available)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")


def chunk_text(text: str, max_chars=4000):
    """
    Split text into chunks of max_chars to avoid tokenizer/translator limits.
    """
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
