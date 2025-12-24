import subprocess

def ask_ollama(model: str, prompt: str) -> str:
    """Send a prompt to an Ollama model and return output text."""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return result.stdout.decode("utf-8").strip()
