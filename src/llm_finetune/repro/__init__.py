"""M6 reproducibility: run registry, model card, and an eval-as-CI gate.

The claim this project makes ("the fine-tune's quality is X, with this
uncertainty") is only credible if a stranger can reproduce it from what's
committed. This package makes that concrete:

* `version.py`  — a deterministic `run_id` derived from the config + the data,
  so the same inputs always name the same run.
* `registry.py` — an append-only registry of runs, each reproducible from its
  recorded config fingerprint.
* `model_card.py` — assembles `MODEL_CARD.md` from the committed artifacts.
* `gate.py` — eval-as-CI: a candidate that regresses vs the promoted checkpoint
  fails the build.
"""
