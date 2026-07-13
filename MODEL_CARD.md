# Model card — Qwen/Qwen2.5-3B-Instruct (domain-QA LoRA)

> Generated from committed artifacts by `python -m llm_finetune.repro.model_card`. Numbers are read from the eval/export JSON, not hand-written.

## Run

- **Run id:** `738b0c490cd9` (deterministic from config + data version)
- **Git commit:** `d0718e47a058de3666efe166b6376ce3938dd853`
- **Backend:** `mock`
- **Base model:** `Qwen/Qwen2.5-3B-Instruct` (max_seq_len 1024)

## LoRA + training config

- **LoRA:** r=16, alpha=32, dropout=0.05, targets=['q_proj', 'k_proj', 'v_proj', 'o_proj']
- **Training:** 3 epochs, batch 4, grad_accum 4, lr 0.0002
- **Seed:** 42

## Dataset

- **Version (content hash):** `9546b8bb8f92`
- **Examples:** 20
- **Category balance:** account: 1, api: 2, billing: 1, dashboards: 3, data-governance: 3, exports: 1, ingestion: 3, reporting: 2, security: 4
- **Token lengths (mean):** question 7.95, answer 21.25

## Evaluation (base vs fine-tuned, held-out test)

- **Judge:** `heuristic-placeholder` (placeholder pending the validated judge)
- **Test items:** 3
- **Headline (mean judge quality):** base 0.502 → tuned 0.502
- **Δ:** +0.000 (95% CI [+0.000, +0.000])
- **Significance:** p-value `pending-validated-judge`

**Verdict:** no measurable difference — CI includes zero; cannot claim the fine-tune beat the base model

## Quantized artifact (M4)

- **Quant level:** `Q4_K_M`
- **Size:** 268 B (268 bytes)
- **GGUF:** `outputs/model.gguf`

## Known limitations

- **Judge is a lexical placeholder**, not the AI Eval Pipeline's validated judge; the significance verdict is preliminary until that lands.
- **Small synthetic sample** — the bundled dataset is a 20-item demo corpus, not a real domain corpus; deltas are illustrative of the pipeline, not a product claim.
- **Backends:** real QLoRA/MLX training, GGUF export, and non-mock serving are implemented behind runtime guards but exercised on hardware, not in CI.

## Reproduce

```bash
pytest -q                                        # data · leak · eval · quant · serve
python -m llm_finetune.pipeline --config config/qa_domain.yaml
python -m llm_finetune.eval.evaluate --mode mock
python -m llm_finetune.quantize.export --mode mock
python -m llm_finetune.repro.build               # re-registers the run + this card
```

The `run_id` above is recomputed from the config + data version, so a cold clone that regenerates it and gets `738b0c490cd9` is looking at the same run.
