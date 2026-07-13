# LLM Fine-Tune — Domain Q&A

Fine-tune an open-source LLM on a custom **domain Q&A** dataset with parameter-efficient
**LoRA / QLoRA**, then **prove it beat the base model** on a held-out test set — the fine-tune
is the treatment, evaluation is the trial.

> Project #2 of a 7-project AI build sequence (Eval → **Fine-Tune** → Knowledge Graph →
> Secure RAG → Agent+Audit → Multi-Agent → Research Team). It's designed to be scored by the
> AI Eval Pipeline (Project #1) so "the fine-tune improved quality" is a statistical claim with
> a confidence interval, not a gut call.

## Status: M5 — serving ✅ (M0 ✅ · M1 ✅ · M2 ✅ · M3 ✅ · M4 ✅)

The full pipeline runs **offline** on a bundled sample dataset via a **mock backend** — no GPU,
no model download. Data prep does lexical near-duplicate removal and emits a stats report.
Real training is implemented on both backends: **QLoRA (4-bit) on CUDA** via `trl` and **LoRA
on Apple Silicon** via `mlx-lm` (a smoke train of Qwen2.5-3B produced a real MLX adapter +
`run.json`). Evaluation compares base vs fine-tuned on the held-out test set with intrinsic
metrics, a judge (correctness/faithfulness/relevance), and **paired-bootstrap CIs** — and writes
an **honest report that can say the fine-tune did _not_ win** (`reports/eval_report.md`).
M4 then merges the winning LoRA into the base weights, quantizes to a single **GGUF**
artifact (recording its size + quant level), and **re-runs a slice of the M3 eval** on the
quantized model to prove quality didn't collapse versus merged fp16 — gated by a configurable
tolerance (`reports/export_report.md`). M5 serves the result behind a **FastAPI** `POST /generate`
endpoint with a singleton model load and a backend-aware engine (vLLM/transformers on CUDA,
MLX or llama.cpp/GGUF on Mac), packaged in a **Dockerfile** that builds and runs the offline
mock engine with no GPU.

> The judge is currently a lexical **placeholder** for the AI Eval Pipeline's validated judge
> (Project #1, not built yet); the validated judge + final p-value land when that project's M5
> does. Deltas and CIs are real; the significance verdict is flagged preliminary until then.

## The one switch that matters: `backend`

Compute is undecided, so the training backend is a single config field, not a rewrite:

| `backend` | Where it runs | Stack |
|-----------|---------------|-------|
| `mock` | anywhere, offline | writes a marker adapter — used for the dry-run and CI |
| `cuda` | NVIDIA GPU | `transformers` + `peft` + `bitsandbytes` (QLoRA 4-bit) + `trl` |
| `mlx`  | Apple Silicon | `mlx-lm` LoRA |

Data prep, splitting, evaluation, and serving are all backend-agnostic.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements/base.txt      # backend-agnostic + dev tools
pip install -e .

# Run the full pipeline offline (prepare -> split -> mock train)
python -m llm_finetune.pipeline --config config/qa_domain.yaml

# M4: merge -> quantize (GGUF) -> sanity-check, offline (writes reports/export_report.md)
python -m llm_finetune.quantize.export --config config/qa_domain.yaml --mode mock

# M5: serve the model (offline mock engine by default)
uvicorn llm_finetune.serve.api:app --port 8000 &
curl -s -X POST localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"question":"...", "context":"..."}'

# Quality gate
pytest && ruff check . && mypy src
```

### Serving (M5)

`POST /generate` loads the model **once** (singleton) and answers domain questions.
The engine is chosen by env so the same app/image serves any backend:

| env var | values | default |
|---|---|---|
| `LLM_FINETUNE_ENGINE` | `mock` · `mlx` · `cuda` · `gguf` | `mock` |
| `LLM_FINETUNE_ADAPTER` | LoRA adapter dir, or a `.gguf` path (for `gguf`) | `train.output_dir` |
| `LLM_FINETUNE_CONFIG` | config path | `config/qa_domain.yaml` |

On CUDA the engine prefers **vLLM** (falls back to transformers); on Mac use `mlx`
or serve the M4 GGUF with `gguf` (llama.cpp). Containerized (defaults to the mock engine):

```bash
docker build -t llm-finetune .
docker run -p 8000:8000 llm-finetune
# real backend, e.g. Mac/MLX:
#   docker build --build-arg EXTRA=requirements/mac.txt -t llm-finetune .
#   docker run -p 8000:8000 -e LLM_FINETUNE_ENGINE=mlx llm-finetune
```

For real training, also install one backend's extras:

```bash
pip install -r requirements/cuda.txt      # on an NVIDIA GPU box
# or
pip install -r requirements/mac.txt       # on an M-series Mac
```

...then set `backend: cuda` (or `mlx`) in `config/qa_domain.yaml`.

## Layout

```
config/qa_domain.yaml        # model, data, LoRA params, and the backend switch
data/sample/domain_qa.jsonl  # 20-item synthetic domain-QA set (runs cold)
src/llm_finetune/
  config.py                  # typed, validated config loaded from YAML
  schema.py                  # QAExample + strict validation + chat formatting
  data/prepare.py            # clean + exact/near dedup -> processed JSONL
  data/split.py              # seeded, leak-safe train/val/test split
  data/stats.py              # dataset stats: counts, length dist, category balance
  train/backend_base.py      # TrainBackend contract (the swappable seam)
  train/backend_common.py    # shared seeding, chat formatting, run metadata
  train/backend_mock.py      # offline dry-run backend
  train/backend_cuda.py      # QLoRA (4-bit) via trl SFT + peft
  train/backend_mlx.py       # MLX LoRA via mlx-lm
  train/train.py             # backend dispatcher
  eval/metrics.py            # intrinsic metrics: exact/norm match, token-F1, ROUGE-L
  eval/generate.py           # base/tuned answer generation (mock · mlx · cuda)
  eval/judge.py              # correctness/faithfulness/relevance (placeholder judge)
  eval/bootstrap.py          # paired-bootstrap CI on the quality delta
  eval/report.py             # base-vs-tuned report (Δ, CI, verdict) -> md + json
  eval/evaluate.py           # evaluation entrypoint
  quantize/artifact.py       # ExportResult + export.json manifest (size + quant level)
  quantize/exporter.py       # Exporter contract + backend dispatcher
  quantize/backend_mock.py   # offline merge marker + placeholder GGUF
  quantize/backend_cuda.py   # peft merge_and_unload -> llama.cpp GGUF quantize
  quantize/backend_mlx.py    # mlx-lm fuse -> GGUF (-> llama.cpp for smaller quants)
  quantize/llama_cpp.py      # thin wrappers over the external llama.cpp toolchain
  quantize/sanity.py         # quantized-vs-merged quality gate (reuses M3 report)
  quantize/export.py         # merge -> quantize -> sanity-check entrypoint
  serve/engine.py            # InferenceEngine contract + shared prompt seam
  serve/registry.py          # engine selection (mock/mlx/cuda/gguf)
  serve/backend_mock.py      # offline deterministic engine (CI + cold run)
  serve/backend_mlx.py       # mlx-lm serving (singleton load)
  serve/backend_cuda.py      # vLLM (preferred) / transformers serving
  serve/backend_gguf.py      # llama.cpp serving of the M4 GGUF artifact
  serve/api.py               # FastAPI app: POST /generate, GET /health
  pipeline.py                # prepare -> split -> train entrypoint
tests/                       # config, schema, data pipeline, dry-run acceptance
```

## Roadmap

- **M0 — Scaffold** ✅ config, schema, data prep/split, backend abstraction, offline dry-run, tests.
- **M1 — Data pipeline** ✅ lexical near-duplicate detection, category-aware records, dataset stats report.
- **M2 — Fine-tuning** ✅ real QLoRA (cuda, trl) and LoRA (mlx-lm) loops; seeded + versioned `run.json`; MLX smoke train produced a real adapter.
- **M3 — Evaluation** ✅ base vs fine-tuned on held-out test: intrinsic metrics + judge + paired-bootstrap CIs; honest committed report (validated judge + p-value pending eval-pipeline M5).
- **M4 — Optimize/export** ✅ merge LoRA into base, quantize to a single GGUF (size + quant recorded), and a tolerance-gated sanity check that re-runs the M3 eval on the quantized model. Mock path runs fully offline; real merge/quantize on `mlx` (mlx-lm fuse) and `cuda` (peft + llama.cpp) behind runtime guards.
- **M5 — Serving** ✅ FastAPI `POST /generate` with a singleton model load and a backend-aware engine (vLLM→transformers on CUDA, mlx-lm or llama.cpp/GGUF on Mac) + Dockerfile. Mock engine serves offline; real engines behind runtime guards.
- **M6 — Reproducibility** model card, run registry, eval-as-CI gate.
