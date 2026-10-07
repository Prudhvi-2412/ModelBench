from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class InferenceProvider(ABC):
    """Provider interface kept independent from the HTTP layer."""

    @abstractmethod
    async def generate(
        self, prompt: str, *, temperature: float, max_tokens: int, top_p: float
    ) -> str:
        raise NotImplementedError

    async def stream(
        self, prompt: str, *, temperature: float, max_tokens: int, top_p: float
    ) -> AsyncIterator[str]:
        """Yield output fragments; providers can override this for token-level streaming."""
        response = await self.generate(
            prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p
        )
        yield response

    @abstractmethod
    def is_loaded(self) -> bool:
        raise NotImplementedError
