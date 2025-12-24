from transformers import pipeline
from .utils import chunk_text

# Load model once (GPU if available)
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device='cuda',
    framework="pt"
)


def summarize_long_text(text: str):

    # Step 1: chunk into pieces < 3000 chars
    chunks = chunk_text(text, max_chars=3000)

    # Step 2: summarize each chunk independently
    partial_summaries = [
        summarizer(c, max_length=120, min_length=60, do_sample=False)[0]["summary_text"]
        for c in chunks
    ]

    # Step 3: combine & summarize again
    combined = " ".join(partial_summaries)
    final_summary = summarizer(
        combined, max_length=150, min_length=80, do_sample=False
    )[0]["summary_text"]

    return final_summary


def compress(text: str):
    summary = summarize_long_text(text)
    model_size_mb = 440
    return summary, model_size_mb
