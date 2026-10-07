import asyncio
from typing import Any

from .base import EmbeddingProvider


class SentenceTransformersProvider(EmbeddingProvider):
    def __init__(self, model_id: str, device: str = "auto", fallback_dimension: int = 384) -> None:
        self.model_id = model_id
        self.device = device
        self.fallback_dimension = fallback_dimension
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Embedding dependencies are missing. Install with: "
                "python -m pip install -e 'backend[embeddings]'"
            ) from exc
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(self.model_id, device=device)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        self._load()
        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vectors.tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, texts)

    def dimension(self) -> int:
        if self._model is None:
            return self.fallback_dimension
        return int(self._model.get_sentence_embedding_dimension())

    def is_loaded(self) -> bool:
        return self._model is not None
