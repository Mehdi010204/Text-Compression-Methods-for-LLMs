from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from .utils import chunk_text

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Load NLLB -----
model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Language codes for NLLB
EN = "eng_Latn"
ZH = "zho_Hans"


def translate(text: str, src: str, tgt: str, max_len=512):
    """
    Chunk-safe translation using NLLB with correct forced BOS token handling.
    """
    chunks = chunk_text(text, max_chars=2000)
    outputs = []

    # Get BOS token ID for target language
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt)

    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        ).to(device)

        output = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_id,
            max_length=max_len,
        )

        translated = tokenizer.decode(output[0], skip_special_tokens=True)
        outputs.append(translated)

    return " ".join(outputs)


def compress(text: str):
    """
    EN -> ZH -> EN cross-lingual compression.
    """
    # English -> Chinese
    zh = translate(text, EN, ZH)

    # Chinese -> English
    back_en = translate(zh, ZH, EN)

    model_size_mb = 600  # approximate model size
    return back_en, model_size_mb
