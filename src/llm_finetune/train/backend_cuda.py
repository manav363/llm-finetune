"""CUDA backend: QLoRA fine-tuning via transformers + peft + bitsandbytes + trl.

Loads the base model in 4-bit (NF4), attaches LoRA adapters, and runs a `trl`
SFT loop over the chat-formatted split. Heavy imports are deferred to call time
so the package still imports on a machine without torch/CUDA (CI, a Mac); on
such a machine `train()` raises `BackendUnavailable` before doing any work.

This loop cannot run in the offline test suite (it needs a CUDA device and a
model download). Its testable seams — seeding, chat formatting, step budget,
run metadata — live in `backend_common` and are covered there.
"""

from __future__ import annotations

from pathlib import Path

from llm_finetune.config import Config
from llm_finetune.train import backend_common as common
from llm_finetune.train.backend_base import BackendUnavailable, TrainResult

_PACKAGES = ["torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate"]


class CudaBackend:
    name = "cuda"

    def train(self, config: Config, train_path: Path, val_path: Path) -> TrainResult:
        try:
            import torch
            from datasets import Dataset
            from peft import LoraConfig
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
            from trl import SFTConfig, SFTTrainer
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise BackendUnavailable(
                "cuda backend requires the CUDA extras: pip install -r requirements/cuda.txt"
            ) from exc

        if not torch.cuda.is_available():  # pragma: no cover - env-dependent
            raise BackendUnavailable("cuda backend selected but no CUDA device is available")

        common.set_all_seeds(config.seed)

        train_examples = common.load_split(train_path)
        val_examples = common.load_split(val_path)
        steps = common.training_iters(
            len(train_examples),
            config.train.batch_size,
            config.train.epochs,
            config.train.max_steps,
        )

        adapter_dir = Path(config.train.output_dir)
        adapter_dir.mkdir(parents=True, exist_ok=True)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model.name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model.config.use_cache = False

        lora = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=list(config.lora.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )

        train_ds = Dataset.from_list(common.to_chat_records(train_examples))
        eval_ds = Dataset.from_list(common.to_chat_records(val_examples))

        sft_config = SFTConfig(
            output_dir=str(adapter_dir),
            num_train_epochs=config.train.epochs,
            max_steps=config.train.max_steps if config.train.max_steps > 0 else -1,
            per_device_train_batch_size=config.train.batch_size,
            gradient_accumulation_steps=config.train.grad_accum,
            learning_rate=config.train.learning_rate,
            max_seq_length=config.model.max_seq_len,
            logging_steps=1,
            save_strategy="epoch",
            seed=config.seed,
            report_to=["wandb"] if config.wandb.enabled else ["none"],
            bf16=True,
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=lora,
            processing_class=tokenizer,
        )
        trainer.train()
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        actual_steps = int(getattr(trainer.state, "global_step", steps) or steps)
        common.write_run_metadata(
            config,
            backend=self.name,
            steps=actual_steps,
            n_train=len(train_examples),
            n_val=len(val_examples),
            package_names=_PACKAGES,
            path=adapter_dir / "run.json",
        )

        return TrainResult(
            backend=self.name,
            adapter_dir=adapter_dir,
            n_train=len(train_examples),
            n_val=len(val_examples),
            steps=actual_steps,
            note=f"QLoRA (4-bit NF4) adapter saved to {adapter_dir}",
        )
