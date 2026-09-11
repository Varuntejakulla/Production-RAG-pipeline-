# app/core/llm_client.py
from typing import AsyncIterator
import httpx
from app.core.config import get_settings


class LLMClient:
    """
    Abstraction over LLM providers. Defaults to Ollama (fully open-source,
    runs locally). Can swap to any OpenAI-compatible endpoint.
    Why abstraction: Lets you change from Llama to Mistral to Qwen
    without touching generation logic.
    """

    def __init__(self):
        settings = get_settings()
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL
        self.base_url = settings.LLM_BASE_URL.rstrip("/")
        self.temperature = settings.LLM_TEMPERATURE
        self.max_tokens = settings.LLM_MAX_TOKENS

    async def generate(self, prompt: str) -> str:
        """Non-streaming generation."""
        if self.provider == "ollama":
            return await self._generate_ollama(prompt)
        else:
            return await self._generate_openai_compatible(prompt)

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Streaming generation — returns tokens as they are produced."""
        if self.provider == "ollama":
            async for token in self._stream_ollama(prompt):
                yield token
        else:
            async for token in self._stream_openai_compatible(prompt):
                yield token

    async def _generate_ollama(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
            )
            resp.raise_for_status()
            return resp.json()["response"]

    async def _stream_ollama(self, prompt: str) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]

    async def _generate_openai_compatible(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]