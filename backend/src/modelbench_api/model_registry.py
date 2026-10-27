from dataclasses import dataclass

from .providers.base import InferenceProvider
from .providers.factory import create_provider
from .settings import ROOT, ModelSettings, load_settings


@dataclass
class ModelRuntime:
    settings: ModelSettings
    provider: InferenceProvider


class ModelRegistry:
    """Resolve named model profiles without mutating a process-wide active model."""

    def __init__(self, default_settings: ModelSettings, default_provider: InferenceProvider):
        self.default = ModelRuntime(default_settings, default_provider)
        self._profiles: dict[str, ModelRuntime] = {}
        self._runtimes: list[ModelRuntime] = []
        self._register(self.default)
        profiles_dir = ROOT / "configs" / "models" / "profiles"
        if profiles_dir.is_dir():
            for config_path in sorted(profiles_dir.glob("*.yaml")):
                settings = load_settings(config_path)
                # The selected default provider owns its model ID. Another profile
                # can use that same ID with a different provider (for example local
                # Transformers and remote vLLM), but the registry can expose only
                # one runtime by model ID in a process. Keep the selected default.
                if settings.model_id == default_settings.model_id:
                    continue
                self._register(ModelRuntime(settings, create_provider(settings)))

    def _register(self, runtime: ModelRuntime) -> None:
        for key in (runtime.settings.name, runtime.settings.model_id):
            if key in self._profiles:
                raise ValueError(f"Duplicate model profile name or ID: {key}")
            self._profiles[key] = runtime
        self._runtimes.append(runtime)

    def resolve(self, name: str | None = None) -> ModelRuntime:
        if name is None:
            return self.default
        try:
            return self._profiles[name]
        except KeyError as exc:
            raise KeyError(f"Model not configured: {name}") from exc

    def all(self) -> list[ModelRuntime]:
        return list(self._runtimes)
