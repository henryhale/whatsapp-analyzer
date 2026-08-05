"""Small local language model, run through ONNX Runtime.

Deliberately no PyTorch. SmolLM2 publishes ONNX weights, so generation runs on
`onnxruntime` via `optimum`, which keeps the whole project inside Streamlit
Community Cloud's memory budget — a torch install alone would eat most of it.

Three tiers, tried in order, so the app degrades instead of crashing:

    360M INT8  (~360 MB)  default
    135M INT8  (~135 MB)  for tighter hosts
    none                  retrieval only, no generation

Be realistic about what a 360M model can do. It will answer a direct question
from retrieved context. It will not reason. Every feature that uses it is built
to stay useful when it is switched off.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

TIERS = [
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "SmolLM2 360M", "~360 MB"),
    ("HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2 135M", "~135 MB"),
]
DEFAULT_MODEL = TIERS[0][0]

# Quantized ONNX weights inside the repo's onnx/ folder, in preference order.
_ONNX_FILES = ["model_q4.onnx", "model_int8.onnx", "model_quantized.onnx", "model.onnx"]


@dataclass
class LLMStatus:
    available: bool
    model_id: str | None = None
    label: str = "Retrieval only"
    detail: str = ""
    error: str | None = None


class LocalLLM:
    """Lazy wrapper around an ONNX causal LM."""

    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        self._model = None
        self._tok = None
        self.status = LLMStatus(available=False)

    # -- loading --

    def load(self) -> LLMStatus:
        if self._model is not None:
            return self.status
        try:
            from optimum.onnxruntime import ORTModelForCausalLM
            from transformers import AutoTokenizer
        except ImportError as e:
            self.status = LLMStatus(
                available=False,
                error=("optimum[onnxruntime] and transformers are not installed. "
                       f"The app still works without generation. ({e})"),
            )
            return self.status

        last_error = None
        for fname in _ONNX_FILES:
            try:
                self._tok = AutoTokenizer.from_pretrained(self.model_id)
                self._model = ORTModelForCausalLM.from_pretrained(
                    self.model_id, file_name=fname, use_cache=True,
                )
                label = next((l for m, l, _ in TIERS if m == self.model_id), self.model_id)
                size = next((s for m, _, s in TIERS if m == self.model_id), "")
                self.status = LLMStatus(True, self.model_id, label, f"{fname} · {size}")
                return self.status
            except Exception as e:  # noqa: BLE001 - any failure means try the next file
                last_error = e
                continue

        self.status = LLMStatus(
            available=False,
            error=f"Could not load {self.model_id}: {last_error}",
        )
        return self.status

    # -- generation --

    def generate(self, prompt: str, max_new_tokens: int = 220,
                 temperature: float = 0.3) -> str:
        """Chat-format a prompt and return only the newly generated text."""
        if self._model is None and not self.load().available:
            raise RuntimeError(self.status.error or "No language model available.")

        messages = [{"role": "user", "content": prompt}]
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(text, return_tensors="pt")
        out = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
            pad_token_id=self._tok.eos_token_id,
        )
        # Slice off the prompt so only the answer comes back.
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self._tok.decode(new_tokens, skip_special_tokens=True).strip()


def probe(model_id: str = DEFAULT_MODEL) -> LLMStatus:
    """Check whether generation is possible without committing to a download."""
    try:
        import optimum.onnxruntime  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return LLMStatus(
            available=False,
            error="optimum[onnxruntime] not installed — running retrieval-only.",
        )
    if os.environ.get("WA_DISABLE_LLM"):
        return LLMStatus(available=False, error="Disabled via WA_DISABLE_LLM.")
    label = next((l for m, l, _ in TIERS if m == model_id), model_id)
    size = next((s for m, _, s in TIERS if m == model_id), "")
    return LLMStatus(True, model_id, label, f"{size} · downloads on first use")
