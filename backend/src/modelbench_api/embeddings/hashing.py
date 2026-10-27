import asyncio
import hashlib
import math
import re
from itertools import pairwise

from .base import EmbeddingProvider


class HashingEmbeddingProvider(EmbeddingProvider):
    """Offline lexical feature hashing for API development, not semantic search."""

    def __init__(self, dimensions: int = 384) -> None:
        self._dimensions = dimensions

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        terms = re.findall(r"\w+", text.lower())
        # Add unigrams and adjacent bigrams; this remains a lexical baseline.
        features = terms + [f"{left}_{right}" for left, right in pairwise(terms)]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vector[value % self._dimensions] += 1.0 if value & (1 << 63) else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(lambda: [self._embed_one(text) for text in texts])

    def dimension(self) -> int:
        return self._dimensions

    def is_loaded(self) -> bool:
        return True
