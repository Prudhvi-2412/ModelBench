from .base import EmbeddingProvider
from .hashing import HashingEmbeddingProvider
from .sentence_transformers import SentenceTransformersProvider
from ..embedding_settings import EmbeddingSettings


def create_embedding_provider(settings: EmbeddingSettings) -> EmbeddingProvider:
    if settings.provider == "hashing":
        return HashingEmbeddingProvider(settings.dimension)
    if settings.provider == "sentence-transformers":
        return SentenceTransformersProvider(settings.model_id, settings.device, settings.dimension)
    raise ValueError(f"Unsupported embedding provider: {settings.provider}")
