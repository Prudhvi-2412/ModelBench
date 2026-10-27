import asyncio
import threading
import time
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
        self.resolved_device: str | None = None
        self._generation_lock = threading.Lock()

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
        self.resolved_device = chosen_device
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self._model.to(chosen_device)
        self._model.eval()

    def _prepare_inputs(self, prompt: str):
        if getattr(self._tokenizer, "chat_template", None):
            inputs = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            inputs = self._tokenizer(prompt, return_tensors="pt")
        return inputs.to(self._model.device)

    def _generate_sync(
        self, prompt: str, temperature: float, max_tokens: int, top_p: float,
        seed: int | None,
    ) -> str:
        import torch

        with self._generation_lock:
            self._load()
            inputs = self._prepare_inputs(prompt)
            generation: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "top_p": top_p,
                "pad_token_id": self._tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation["temperature"] = temperature
            devices = [self._model.device.index or 0] if self._model.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                if seed is not None:
                    torch.manual_seed(seed)
                    if devices:
                        torch.cuda.manual_seed_all(seed)
                with torch.inference_mode():
                    output = self._model.generate(**inputs, **generation)
            new_tokens = output[0, inputs["input_ids"].shape[1] :]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    async def generate(
        self, prompt: str, *, temperature: float, max_tokens: int, top_p: float,
        seed: int | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._generate_sync, prompt, temperature, max_tokens, top_p, seed
        )

    async def prepare(self) -> float:
        if self.is_loaded():
            return 0.0
        started = time.perf_counter()
        await asyncio.to_thread(self._load_under_lock)
        return round((time.perf_counter() - started) * 1000, 3)

    def _load_under_lock(self) -> None:
        with self._generation_lock:
            self._load()

    def _stream_sync(
        self, prompt: str, temperature: float, max_tokens: int, top_p: float,
        seed: int | None,
    ):
        import torch
        from transformers import TextIteratorStreamer

        with self._generation_lock:
            self._load()
            inputs = self._prepare_inputs(prompt)
        generation: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation["temperature"] = temperature
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        def run_generation() -> None:
            devices = [self._model.device.index or 0] if self._model.device.type == "cuda" else []
            with self._generation_lock, torch.random.fork_rng(devices=devices):
                if seed is not None:
                    torch.manual_seed(seed)
                    if devices:
                        torch.cuda.manual_seed_all(seed)
                with torch.inference_mode():
                    self._model.generate(**inputs, **generation, streamer=streamer)

        worker = threading.Thread(target=run_generation, daemon=True)
        worker.start()
        return iter(streamer)

    async def stream(
        self, prompt: str, *, temperature: float, max_tokens: int, top_p: float,
        seed: int | None = None,
    ) -> AsyncIterator[str]:
        iterator = await asyncio.to_thread(
            self._stream_sync, prompt, temperature, max_tokens, top_p, seed
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
