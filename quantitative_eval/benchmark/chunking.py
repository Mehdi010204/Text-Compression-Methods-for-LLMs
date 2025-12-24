import re
from sentence_transformers import util
from compression.utils import embedder, chunk_text


def encode_long_text(text: str):
    # Split into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) == 0:
        return embedder.encode("empty", convert_to_tensor=True)

    # Encode each sentence individually (safe)
    embeddings = embedder.encode(sentences, convert_to_tensor=True)
    return embeddings.mean(dim=0)  # average pooling
