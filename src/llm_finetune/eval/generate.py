"""Answer generation for the same held-out items — base and fine-tuned.

A `Generator` maps test examples to answers at temperature 0 (greedy, so the
snapshot is deterministic). The mock generator runs offline for CI; the MLX and
CUDA generators load the real base model, optionally with the M2 LoRA adapter,
and defer heavy imports so the package still imports without them.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from llm_finetune.schema import QAExample
from llm_finetune.train.backend_base import BackendUnavailable

# Generation is greedy so runs are reproducible and comparable across models.
GREEDY_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 256


@dataclass(frozen=True)
class Generation:
    """One generated answer alongside the item it answers."""

    id: str
    question: str
    context: str
    reference: str
    answer: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class Generator(Protocol):
    name: str

    def generate(self, examples: list[QAExample]) -> list[Generation]:
        ...


def _to_generation(example: QAExample, answer: str) -> Generation:
    return Generation(
        id=example.id,
        question=example.question,
        context=example.context,
        reference=example.answer,
        answer=answer.strip(),
    )


class MockGenerator:
    """Deterministic offline generator driven by a pure transform.

    Defaults to echoing the reference answer (a "perfect" mock). Tests and the
    mock-mode demo supply their own transform to model weaker/stronger answers.
    """

    def __init__(
        self,
        *,
        name: str = "mock",
        transform: Callable[[QAExample], str] | None = None,
    ) -> None:
        self.name = name
        self._transform = transform or (lambda ex: ex.answer)

    def generate(self, examples: list[QAExample]) -> list[Generation]:
        return [_to_generation(ex, self._transform(ex)) for ex in examples]


class MlxGenerator:
    """Greedy generation via mlx-lm, optionally with a LoRA adapter."""

    def __init__(
        self,
        model_name: str,
        *,
        adapter_path: Path | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        name: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.max_tokens = max_tokens
        self.name = name or ("mlx-tuned" if adapter_path else "mlx-base")

    def generate(self, examples: list[QAExample]) -> list[Generation]:
        try:
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise BackendUnavailable(
                "mlx generation requires the Apple-Silicon extras: "
                "pip install -r requirements/mac.txt"
            ) from exc

        adapter = str(self.adapter_path) if self.adapter_path else None
        model, tokenizer = load(self.model_name, adapter_path=adapter)
        sampler = make_sampler(temp=GREEDY_TEMPERATURE)

        results: list[Generation] = []
        for ex in examples:
            prompt = tokenizer.apply_chat_template(
                ex.to_prompt_messages(), add_generation_prompt=True, tokenize=False
            )
            text = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                sampler=sampler,
                verbose=False,
            )
            results.append(_to_generation(ex, text))
        return results


class CudaGenerator:
    """Greedy generation via transformers, optionally with a PEFT adapter."""

    def __init__(
        self,
        model_name: str,
        *,
        adapter_path: Path | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        name: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.max_tokens = max_tokens
        self.name = name or ("cuda-tuned" if adapter_path else "cuda-base")

    def generate(self, examples: list[QAExample]) -> list[Generation]:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise BackendUnavailable(
                "cuda generation requires the CUDA extras: "
                "pip install -r requirements/cuda.txt"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(self.adapter_path))
        model.eval()

        results: list[Generation] = []
        for ex in examples:
            prompt = tokenizer.apply_chat_template(
                ex.to_prompt_messages(), add_generation_prompt=True, tokenize=False
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=self.max_tokens, do_sample=False
                )
            text = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            results.append(_to_generation(ex, text))
        return results


def write_generations(generations: list[Generation], path: str | Path) -> Path:
    """Snapshot generations to JSONL (one object per line)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for gen in generations:
            fh.write(json.dumps(gen.to_dict(), ensure_ascii=False) + "\n")
    return out
