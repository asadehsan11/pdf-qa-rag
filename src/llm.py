"""Gemini helper with retry and model fallback."""
import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]

from google.genai import types

_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=20000),
)


def generate(prompt: str, retries: int = 1) -> str:
    """Try each model in order; retry briefly on temporary errors."""
    last_error = None
    for model in MODELS:
        for attempt in range(retries):
            try:
                response = _client.models.generate_content(model=model, contents=prompt)
                return response.text
            except Exception as e:
                last_error = e
                print(f"{model} failed ({type(e).__name__}), trying again...")
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"All models failed: {last_error}")


if __name__ == "__main__":
    print(generate("Say hello in one short sentence."))