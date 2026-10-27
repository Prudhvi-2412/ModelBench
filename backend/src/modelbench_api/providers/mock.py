from collections.abc import AsyncIterator

from .base import InferenceProvider


class DevelopmentProvider(InferenceProvider):
    """No-download provider for developing and exploring the API contract."""

    async def generate(
        self, prompt: str, *, temperature: float, max_tokens: int, top_p: float,
        seed: int | None = None,
    ) -> str:
        del temperature, max_tokens, top_p, seed
        excerpt = " ".join(prompt.split())[:240]
        return (
            "Development provider response (this is not an LLM completion). "
            f"Received prompt: {excerpt}"
        )

    async def stream(
        self, prompt: str, *, temperature: float, max_tokens: int, top_p: float,
        seed: int | None = None,
    ) -> AsyncIterator[str]:
        response = await self.generate(
            prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p, seed=seed
        )
        words = response.split(" ")
        for index, word in enumerate(words):
            yield word if index == 0 else f" {word}"

    def is_loaded(self) -> bool:
        return True
