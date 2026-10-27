import json
from collections.abc import AsyncIterator

import httpx

from .base import InferenceProvider


class VllmProvider(InferenceProvider):
    """OpenAI-compatible adapter for a separately served vLLM endpoint."""

    def __init__(self, model_id: str, base_url: str, api_key: str | None = None) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._ready = False

    async def prepare(self) -> None:
        """Check that the vLLM API is serving before a benchmark begins.

        Remote model load time is not observable through this adapter, so it is
        intentionally not reported as this process's cold-load duration.
        """
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/models", headers=self._headers())
                response.raise_for_status()
                models = response.json().get("data", [])
                if not models:
                    raise RuntimeError("vLLM readiness endpoint returned no model entries")
                if not any(item.get("id") == self.model_id for item in models):
                    available = ", ".join(item.get("id", "?") for item in models)
                    raise RuntimeError(
                        f"vLLM is ready but does not serve {self.model_id}; available: {available}"
                    )
                self._ready = True
        except httpx.HTTPError as exc:
            self._ready = False
            raise RuntimeError(f"vLLM is not ready at {self.base_url}: {exc}") from exc

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def generate(
        self, prompt: str, *, temperature: float, max_tokens: int, top_p: float,
        seed: int | None = None,
    ) -> str:
        payload = {"model": self.model_id, "messages": [{"role": "user", "content": prompt}],
                   "temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}
        if seed is not None:
            payload["seed"] = seed
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers())
                response.raise_for_status()
                self._ready = True
                return response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise RuntimeError(f"vLLM request failed at {self.base_url}: {exc}") from exc

    async def stream(
        self, prompt: str, *, temperature: float, max_tokens: int, top_p: float,
        seed: int | None = None,
    ) -> AsyncIterator[str]:
        payload = {"model": self.model_id, "messages": [{"role": "user", "content": prompt}],
                   "temperature": temperature, "max_tokens": max_tokens, "top_p": top_p, "stream": True}
        if seed is not None:
            payload["seed"] = seed
        try:
            async with (
                httpx.AsyncClient(timeout=300) as client,
                client.stream("POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers()) as response,
            ):
                response.raise_for_status()
                self._ready = True
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        yield delta
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            raise RuntimeError(f"vLLM stream failed at {self.base_url}: {exc}") from exc

    def is_loaded(self) -> bool:
        return self._ready
