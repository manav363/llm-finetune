"""Clean-checkout end-to-end test of the documented CLI command flow.

This guards the exact sequence CI runs — `data.prepare` -> `data.split` ->
`train.train` -> `eval.evaluate` -> `repro.gate` — by invoking each as a real
subprocess module (`python -m ...`). A module missing its `main()` (the bug that
silently broke CI) fails here because the step no-ops and `test.jsonl` is never
written. Everything runs offline on the bundled sample via the mock backend.
"""

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE = REPO_ROOT / "data" / "sample" / "domain_qa.jsonl"


def _write_config(tmp_path: Path) -> Path:
    config = {
        "seed": 42,
        "backend": "mock",
        "model": {"name": "Qwen/Qwen2.5-3B-Instruct", "max_seq_len": 512},
        "data": {
            "raw_path": str(SAMPLE),
            "processed_path": str(tmp_path / "processed" / "domain_qa.jsonl"),
            "splits_dir": str(tmp_path / "processed" / "splits"),
            "val_frac": 0.15,
            "test_frac": 0.15,
            "near_dup_threshold": 0.85,
        },
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
        "train": {
            "epochs": 1, "batch_size": 4, "grad_accum": 1, "learning_rate": 0.0002,
            "output_dir": str(tmp_path / "adapter"), "max_steps": 0,
        },
        "wandb": {"enabled": False, "project": "llm-finetune"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"{module} failed:\n{result.stderr}"
    return result


def test_cli_flow_produces_splits_eval_and_passes_gate(tmp_path):
    cfg = str(_write_config(tmp_path))
    reports = tmp_path / "reports"

    _run("llm_finetune.data.prepare", "--config", cfg)
    _run("llm_finetune.data.split", "--config", cfg)
    _run("llm_finetune.train.train", "--config", cfg)

    # The bug that broke CI: split no-ops -> no test.jsonl -> eval crashes.
    test_split = tmp_path / "processed" / "splits" / "test.jsonl"
    assert test_split.is_file(), "split step did not produce test.jsonl"

    _run("llm_finetune.eval.evaluate", "--config", cfg, "--mode", "mock",
         "--out-dir", str(reports))
    assert (reports / "eval_report.json").is_file()

    # Gate a report against itself -> no regression -> exit 0.
    gate = subprocess.run(
        [sys.executable, "-m", "llm_finetune.repro.gate",
         "--candidate", str(reports / "eval_report.json"),
         "--promoted", str(reports / "eval_report.json")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert gate.returncode == 0, gate.stdout + gate.stderr
