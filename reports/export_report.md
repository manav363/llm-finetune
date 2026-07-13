# M4 — optimize & export

- **Backend:** `mock`
- **Merged fp16:** `outputs/merged`
- **Quantized artifact:** `outputs/model.gguf` (Q4_K_M, 268 B)
- **Note:** mock export: wrote merged.json marker + placeholder GGUF, no quantization

## Quantization sanity check

- **Quant level:** `Q4_K_M`
- **Test-slice items:** 3
- **Tolerance (max headline drop):** 0.050
- **Merged fp16 (mean judge quality):** 0.655
- **Quantized (mean judge quality):** 0.655
- **Δ (quantized − merged):** +0.000

**PASS — quantized (Q4_K_M) holds quality within tolerance: Δ +0.000 ≥ -0.050**
