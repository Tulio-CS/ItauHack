"""Utility helpers for running a lightweight instruction-tuned model locally.

The functions defined in this module wrap a HuggingFace ``transformers`` model so
that we can reuse the same instance across different prompts (relevance
classification, structured extraction and impact scoring).  The module prefers a
small T5 model by default which fits in CPU only environments while still
producing good quality structured responses when guided with constrained
prompts.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class GenerationConfig:
    """Configuration for the local LLM generation."""

    model_name: str = "google/flan-t5-base"
    max_new_tokens: int = 256
    temperature: float = 0.0
    repetition_penalty: float = 1.05


class LocalLLM:
    """Simple wrapper around a seq2seq LLM model running locally.

    Parameters
    ----------
    model_name:
        Hugging Face identifier of the model.  Defaults to
        ``google/flan-t5-base`` which is instruction tuned and relatively
        lightweight so it runs on CPU.
    device:
        Target device passed to ``transformers``.  ``None`` lets the library
        decide (typically CPU inside this challenge environment).
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> None:
        self.config = generation_config or GenerationConfig()
        if model_name:
            self.config.model_name = model_name
        self._device = device

        self._tokenizer = get_tokenizer(self.config.model_name)
        self._model = get_model(self.config.model_name)
        if device:
            self._model.to(device)

    def generate(self, prompt: str, **kwargs: Dict) -> str:
        """Generate a deterministic response for ``prompt``.

        ``kwargs`` can override the default generation hyper-parameters on a per
        call basis.
        """

        inputs = self._tokenizer(prompt, return_tensors="pt")
        config = {**self.config.__dict__, **kwargs}
        generated_tokens = self._model.generate(
            **inputs,
            max_new_tokens=config.get("max_new_tokens", self.config.max_new_tokens),
            temperature=config.get("temperature", self.config.temperature),
            repetition_penalty=config.get(
                "repetition_penalty", self.config.repetition_penalty
            ),
        )
        return self._tokenizer.decode(generated_tokens[0], skip_special_tokens=True)


@lru_cache(maxsize=2)
def get_tokenizer(model_name: str):
    """Load and cache the tokenizer."""

    return AutoTokenizer.from_pretrained(model_name)


@lru_cache(maxsize=2)
def get_model(model_name: str):
    """Load and cache the seq2seq model."""

    return AutoModelForSeq2SeqLM.from_pretrained(model_name)
