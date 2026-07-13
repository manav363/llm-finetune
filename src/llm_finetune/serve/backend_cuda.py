"""CUDA engine: NVIDIA serving, high-throughput vLLM when available.

Prefers vLLM (paged-attention, fast batched decode); falls back to a plain
transformers generate loop when vLLM isn't installed. Either way the model is
loaded once and reused. vLLM serves a merged checkpoint, so a LoRA adapter is
applied via transformers in the fallback path or expected pre-merged (M4) for
vLLM. Heavy imports are deferred and guarded; on a CPU-only / CI box `load`
raises `BackendUnavailable`. Real serving awaits a GPU box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_finetune.serve.engine import DEFAULT_MAX_TOKENS, GREEDY_TEMPERATURE, build_messages
from llm_finetune.train.backend_base import BackendUnavailable


class CudaEngine:
    name = "cuda"

    def __init__(self, model_name: str, *, adapter: Path | None = None) -> None:
        self.model_name = model_name
        self.adapter = adapter
        self._backend = ""  # "vllm" or "transformers", set at load
        self._llm: Any = None
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        # vLLM is the preferred serving path; it can't apply a LoRA adapter
        # inline here, so use it only for a pre-merged checkpoint (M4 output).
        if self.adapter is None and self._try_load_vllm():
            return
        self._load_transformers()

    def _try_load_vllm(self) -> bool:
        try:
            from vllm import LLM
        except ImportError:
            return False
        self._llm = LLM(model=self.model_name)  # pragma: no cover - env-dependent
        self._backend = "vllm"  # pragma: no cover - env-dependent
        return True  # pragma: no cover - env-dependent

    def _load_transformers(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise BackendUnavailable(
                "cuda serving requires the CUDA extras: pip install -r requirements/cuda.txt"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(  # pragma: no cover - env-dependent
            self.model_name
        )
        model = AutoModelForCausalLM.from_pretrained(  # pragma: no cover - env-dependent
            self.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        if self.adapter:  # pragma: no cover - env-dependent
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(self.adapter))
        model.eval()  # pragma: no cover - env-dependent
        self._model = model  # pragma: no cover - env-dependent
        self._backend = "transformers"  # pragma: no cover - env-dependent

    def generate(
        self,
        question: str,
        context: str = "",
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = GREEDY_TEMPERATURE,
    ) -> str:
        if not self._backend:
            self.load()
        messages = build_messages(question, context)
        if self._backend == "vllm":  # pragma: no cover - env-dependent
            from vllm import SamplingParams

            out = self._llm.chat(
                messages,
                SamplingParams(temperature=temperature, max_tokens=max_tokens),
            )
            text: str = out[0].outputs[0].text
            return text.strip()
        return self._generate_transformers(messages, max_tokens, temperature)

    def _generate_transformers(
        self, messages: list[dict[str, str]], max_tokens: int, temperature: float
    ) -> str:  # pragma: no cover - env-dependent
        import torch

        prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=temperature > 0.0
            )
        decoded: str = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return decoded.strip()
