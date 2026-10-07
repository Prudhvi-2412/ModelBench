import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any

from .base import InferenceProvider


class TransformersProvider(InferenceProvider):
    """Lazily load a Hugging Face causal language model on CPU or CUDA."""

    def __init__(self, model_id: str, device: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers provider dependencies are missing. Install with: "
                "python -m pip install -e 'backend[transformers]'"
            ) from exc

        chosen_device = self.device
        if chosen_device == "auto":
            chosen_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self._model.to(chosen_device)
        self._model.eval()

    def _generate_sync(self, prompt: str, temperature: float, max_tokens: int, top_p: float) -> str:
        import torch

        self._load()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        generation: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation["temperature"] = temperature
        with torch.inference_mode():
            output = self._model.generate(**inputs, **generation)
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    async def generate(
        self, prompt: str, *, temperature: float, max_tokens: int, top_p: float
    ) -> str:
        return await asyncio.to_thread(
            self._generate_sync, prompt, temperature, max_tokens, top_p
        )

    def _stream_sync(self, prompt: str, temperature: float, max_tokens: int, top_p: float):
        from transformers import TextIteratorStreamer

        self._load()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        generation: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation["temperature"] = temperature
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        worker = threading.Thread(
            target=self._model.generate,
            kwargs={**inputs, **generation, "streamer": streamer},
            daemon=True,
        )
        worker.start()
        return iter(streamer)

    async def stream(
        self, prompt: str, *, temperature: float, max_tokens: int, top_p: float
    ) -> AsyncIterator[str]:
        iterator = await asyncio.to_thread(
            self._stream_sync, prompt, temperature, max_tokens, top_p
        )
        sentinel = object()

        def next_fragment():
            try:
                return next(iterator)
            except StopIteration:
                return sentinel

        while True:
            fragment = await asyncio.to_thread(next_fragment)
            if fragment is sentinel:
                break
            if fragment:
                yield fragment

    def is_loaded(self) -> bool:
        return self._model is not None
