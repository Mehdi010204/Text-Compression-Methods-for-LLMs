import re
import numpy as np
from sentence_transformers import util
from .utils import embedder   

def compress(text: str, keep_ratio: float = 0.4, min_sentences: int = 5):

    # --- Split into sentences ---
    sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]
    if len(sentences) == 0:
        return text, 90  # fail-safe

    # Query used to measure importance
    query = "What is the main idea of the text?"

    # Encode sentences & query
    emb_sents = embedder.encode(sentences, convert_to_tensor=True)
    emb_query = embedder.encode(query, convert_to_tensor=True)

    # Compute cosine similarity for each sentence
    scores = util.cos_sim(emb_query, emb_sents)[0].cpu().numpy()

    # How many sentences to keep
    top_k = max(min_sentences, int(len(sentences) * keep_ratio))

    # Choose most relevant sentences
    top_idx = np.argsort(-scores)[:top_k]
    selected = [sentences[i] for i in sorted(top_idx)]

    compressed_text = " ".join(selected)
    model_size_mb = 90  # assumed for reporting

    return compressed_text, model_size_mb
