from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class InferenceProvider(ABC):
    """Provider interface kept independent from the HTTP layer."""

    @abstractmethod
    async def generate(
        self, prompt: str, *, temperature: float, max_tokens: int, top_p: float,
        seed: int | None = None,
    ) -> str:
        raise NotImplementedError

    async def stream(
        self, prompt: str, *, temperature: float, max_tokens: int, top_p: float,
        seed: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield output fragments; providers can override this for token-level streaming."""
        response = await self.generate(
            prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p, seed=seed
        )
        yield response

    async def prepare(self) -> float | None:
        """Prepare lazy resources before timed inference, if the provider supports it."""
        return None

    @abstractmethod
    def is_loaded(self) -> bool:
        raise NotImplementedError
