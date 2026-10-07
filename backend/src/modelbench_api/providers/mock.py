from .base import InferenceProvider
from collections.abc import AsyncIterator


class DevelopmentProvider(InferenceProvider):
    """No-download provider for developing and exploring the API contract."""

    async def generate(
        self, prompt: str, *, temperature: float, max_tokens: int, top_p: float
    ) -> str:
        del temperature, max_tokens, top_p
        excerpt = " ".join(prompt.split())[:240]
        return (
            "Development provider response (this is not an LLM completion). "
            f"Received prompt: {excerpt}"
        )

    async def stream(
        self, prompt: str, *, temperature: float, max_tokens: int, top_p: float
    ) -> AsyncIterator[str]:
        response = await self.generate(
            prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p
        )
        words = response.split(" ")
        for index, word in enumerate(words):
            yield word if index == 0 else f" {word}"

    def is_loaded(self) -> bool:
        return True
