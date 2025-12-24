from sentence_transformers import SentenceTransformer, util

# Load once globally for efficiency
_sim_model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(a: str, b: str) -> float:
    """Returns cosine similarity between embeddings of texts a and b."""
    emb_a = _sim_model.encode(a, convert_to_tensor=True)
    emb_b = _sim_model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(emb_a, emb_b)[0][0])
