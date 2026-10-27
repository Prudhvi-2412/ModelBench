from ..settings import ModelSettings
from .base import InferenceProvider
from .mock import DevelopmentProvider
from .transformers_provider import TransformersProvider
from .vllm_provider import VllmProvider


def create_provider(settings: ModelSettings) -> InferenceProvider:
    if settings.provider == "mock":
        return DevelopmentProvider()
    if settings.provider == "transformers":
        return TransformersProvider(settings.model_id, settings.device)
    if settings.provider == "vllm":
        return VllmProvider(settings.model_id, settings.api_base_url, settings.api_key)
    raise ValueError(f"Unsupported inference provider: {settings.provider}")
