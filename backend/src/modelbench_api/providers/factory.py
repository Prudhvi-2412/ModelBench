from .base import InferenceProvider
from .mock import DevelopmentProvider
from .transformers_provider import TransformersProvider
from ..settings import ModelSettings


def create_provider(settings: ModelSettings) -> InferenceProvider:
    if settings.provider == "mock":
        return DevelopmentProvider()
    if settings.provider == "transformers":
        return TransformersProvider(settings.model_id, settings.device)
    raise ValueError(f"Unsupported inference provider: {settings.provider}")
