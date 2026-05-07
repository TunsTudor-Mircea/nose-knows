"""
SLM loader — thin HTTP wrapper around the Ollama REST API.

Bypasses LangChain's Ollama client entirely to avoid the '_type' serialisation
bug in langchain-community 0.2.x.  The public interface is identical:

    llm = load_llm()
    text = llm.invoke("What perfume notes evoke a rainy forest?")

Configuration (via .env):
    OLLAMA_MODEL    — model tag (default: gemma2:2b)
    OLLAMA_BASE_URL — Ollama server URL (default: http://localhost:11434)
                      In Docker Compose this should be http://ollama:11434
"""
from __future__ import annotations

import os
from functools import lru_cache

import httpx
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_MODEL = "gemma2:2b"
_DEFAULT_BASE_URL = "http://localhost:11434"
_TIMEOUT = 300  # seconds — large models can be slow on first token


class OllamaLLM:
    """Minimal Ollama client with a LangChain-compatible `.invoke()` interface."""

    def __init__(self, model: str, base_url: str, temperature: float = 0.7, num_predict: int = 512):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict

    def invoke(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the generated text."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["response"]

    # Allow the object to be called like a function (some internal code paths use this)
    def __call__(self, prompt: str) -> str:
        return self.invoke(prompt)


@lru_cache(maxsize=1)
def load_llm() -> OllamaLLM:
    """Return a cached OllamaLLM instance."""
    model = os.getenv("OLLAMA_MODEL", _DEFAULT_MODEL)
    base_url = os.getenv("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)
    print(f"[model] Using Ollama at {base_url}, model={model}")
    return OllamaLLM(model=model, base_url=base_url)
