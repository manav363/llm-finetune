# LLM Fine-Tune — Domain Q&A

[![CI](https://img.shields.io/github/actions/workflow/status/manav363/llm-finetune/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/manav363/llm-finetune/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
[![License](https://img.shields.io/badge/license-MIT-00D4AA?style=flat-square)](LICENSE)

A fine-tuning **pipeline** for a domain question-answering model: data preparation,
parameter-efficient training (**LoRA on Apple Silicon via MLX**, **QLoRA on NVIDIA via
trl/peft**), a statistical base-vs-fine-tuned evaluation, GGUF export, a FastAPI serving layer,
and checks that a run can be reproduced. Every stage also runs offline on a bundled sample
through a **mock backend**, so the whole thing can be cloned and exercised with no GPU and no
model download.

The design principle is that "the fine-tune helped" should be a *measured* claim with a
confidence interval, and that the report is allowed to say it did not.

## What this repository does and does not show

| Stage | State | Evidence in this repo |
|---|---|---|
| Data prep: cleaning, near-duplicate removal, seeded leak-safe split, stats report | Implemented | Unit tests, run in CI |
| Training, MLX LoRA | Implemented, **run once as a smoke train** (Qwen2.5-3B on a Mac) | The resulting adapter and `run.json` were not committed (weights are git-ignored) |
| Training, CUDA QLoRA | Implemented behind runtime guards | **No recorded GPU run** in the repository |
| Evaluation: intrinsic metrics, judge, paired-bootstrap CIs | Implemented | Unit tests; one committed real-model report (below) |
| Export: merge LoRA, quantize to GGUF, tolerance-gated sanity check | Implemented | Mock path tested in CI; the committed `reports/export.json` is a **mock placeholder** (a 268-byte marker file, not a quantized model) |
| Serving: FastAPI `POST /generate`, `GET /health` | Implemented | Mock engine tested in CI; real engines (MLX, CUDA/vLLM, llama.cpp) not exercised in CI |
| Reproducibility: deterministic `run_id`, run registry, generated model card, eval gate | Implemented | Unit tests; gate runs in CI |

### The committed result

`reports/eval_report.md` compares base `Qwen/Qwen2.5-3B-Instruct` against the smoke-trained
adapter on a **3-item test split** from a **20-item synthetic dataset**. Every metric is
identical for both models (Δ +0.000, 95% CI [+0.000, +0.000]) and the report's verdict is
*"no measurable difference — cannot claim the fine-tune beat the base model"*.

That is the honest state of the project: the pipeline works end to end, but **this repository
does not demonstrate a model improvement**. Doing so needs a real dataset, a longer training
run, and a validated judge. The 20-item sample exists to exercise the code, not to measure
quality.

> **About the judge.** The judge is currently a lexical **placeholder**. The reports refer to a
> "validated judge" from an *AI Eval Pipeline*; that is a separate project which has not been
> built yet. Until a validated judge is wired in, deltas and confidence intervals are real
> computations but the significance verdict is flagged preliminary.

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

### Reproducibility (M6)

Every run is named by a deterministic `run_id` = hash(reproducibility-relevant config +
content hash of the data). Same inputs → same id on any machine, so a cold clone can
recompute the id in the registry and know it's the same run.

```bash
python -m llm_finetune.repro.build          # register the run + (re)write MODEL_CARD.md
python -m llm_finetune.repro.gate \          # eval-as-CI: exit 1 if candidate regresses
  --candidate reports/eval_report.json --promoted reports/eval_report.json
```

[`MODEL_CARD.md`](MODEL_CARD.md) is generated from the committed eval/export JSON (never
hand-written), and the same gate runs in CI (`.github/workflows/ci.yml`) to block a fine-tune
that regresses versus the promoted checkpoint.

One thing to know about that gate: in CI the candidate report is produced by the **mock**
backend, so the gate there checks that the pipeline still produces a consistent report and that
the comparison logic works. It becomes a quality gate when the candidate report comes from a
real evaluation run (`python -m llm_finetune.eval.evaluate --mode mlx` or `cuda`) and is passed
to `repro.gate` with the promoted report.

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
  repro/version.py           # deterministic data_version + run_id (reproducible from config)
  repro/registry.py          # append-only run registry (RunRecord, upsert by run_id)
  repro/model_card.py        # assemble MODEL_CARD.md from committed artifacts
  repro/gate.py              # eval-as-CI gate: regression vs promoted -> non-zero exit
  repro/build.py             # register the run + (re)write the model card
  pipeline.py                # prepare -> split -> train entrypoint
tests/                       # unit + end-to-end tests for every stage (all run offline)
reports/                     # committed eval/export reports (see "The committed result")
```

## Roadmap

- **M0 — Scaffold** ✅ config, schema, data prep/split, backend abstraction, offline dry-run, tests.
- **M1 — Data pipeline** ✅ lexical near-duplicate detection, category-aware records, dataset stats report.
- **M2 — Fine-tuning** ✅ real QLoRA (cuda, trl) and LoRA (mlx-lm) loops; seeded + versioned `run.json`; MLX smoke train produced a real adapter.
- **M3 — Evaluation** ✅ base vs fine-tuned on held-out test: intrinsic metrics + judge + paired-bootstrap CIs; honest committed report (validated judge + p-value pending eval-pipeline M5).
- **M4 — Optimize/export** ✅ merge LoRA into base, quantize to a single GGUF (size + quant recorded), and a tolerance-gated sanity check that re-runs the M3 eval on the quantized model. Mock path runs fully offline; real merge/quantize on `mlx` (mlx-lm fuse) and `cuda` (peft + llama.cpp) behind runtime guards.
- **M5 — Serving** ✅ FastAPI `POST /generate` with a singleton model load and a backend-aware engine (vLLM→transformers on CUDA, mlx-lm or llama.cpp/GGUF on Mac) + Dockerfile. Mock engine serves offline; real engines behind runtime guards.
- **M6 — Reproducibility** ✅ deterministic `run_id` (config + data version), append-only run registry, generated [`MODEL_CARD.md`](MODEL_CARD.md), and an eval-as-CI gate wired into GitHub Actions that fails the build on a regression vs the promoted eval report.

### Not done yet

- A real (non-synthetic) dataset with a test split large enough for the confidence intervals to mean something.
- A fine-tuning run long enough to move the metrics, evaluated with a validated judge.
- A recorded CUDA/QLoRA run and a real (non-mock) GGUF export committed or published as a release asset.

## How this was built

This project was built with AI coding assistance (Claude). What can be checked without taking
that on trust: the type-checked (`mypy --strict`) and linted source, the offline test suite
(`pytest`), and the committed reports above, which say plainly what has and has not been shown.

## License

[MIT](LICENSE)
