# Evaluation: base vs fine-tuned

- **Base model:** Qwen/Qwen2.5-3B-Instruct
- **Adapter:** outputs/adapter
- **Generation:** base=`mlx-base` · tuned=`mlx-tuned` (greedy, temp=0)
- **Judge:** `heuristic-placeholder`
- **Test items:** 3
- **Significance:** p-value `pending-validated-judge` (local bootstrap CIs shown; validated judge + p-value land with eval-pipeline M5)

## Verdict

**no measurable difference — CI includes zero; cannot claim the fine-tune beat the base model**

Headline (mean judge quality): base 0.502 → tuned 0.502, Δ +0.000 (95% CI [+0.000, +0.000]).

## Judge dimensions

| dimension | base | tuned | Δ | 95% CI | p (prelim) |
|---|---|---|---|---|---|
| correctness | 0.456 | 0.456 | +0.000 | [+0.000, +0.000] | 0.000 |
| faithfulness | 0.446 | 0.446 | +0.000 | [+0.000, +0.000] | 0.000 |
| relevance | 0.604 | 0.604 | +0.000 | [+0.000, +0.000] | 0.000 |

## Intrinsic metrics

| metric | base | tuned | Δ | 95% CI | p (prelim) |
|---|---|---|---|---|---|
| exact_match | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | 0.000 |
| normalized_match | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | 0.000 |
| token_f1 | 0.456 | 0.456 | +0.000 | [+0.000, +0.000] | 0.000 |
| rouge_l | 0.456 | 0.456 | +0.000 | [+0.000, +0.000] | 0.000 |

> The judge here is a lexical **placeholder**, not the AI Eval Pipeline's validated judge. Deltas and CIs are real; the significance verdict is preliminary until the validated judge is wired in.
