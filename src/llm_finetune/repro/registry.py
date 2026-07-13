"""An append-only run registry — one record per training/eval run.

Each `RunRecord` pins everything needed to identify and reproduce a run: its
deterministic `run_id`, the config fingerprint, the data version, the git commit,
pointers to the artifacts it produced, and a compact eval summary. The registry
is JSONL so it appends cleanly and diffs well; `register_run` upserts by
`run_id`, so re-recording the same inputs updates in place rather than
duplicating.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from llm_finetune.config import Config
from llm_finetune.repro import version as ver

DEFAULT_REGISTRY = Path("runs/registry.jsonl")


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    created_at: str
    git_commit: str | None
    backend: str
    base_model: str
    data_version: str
    config_fingerprint: dict[str, object]
    artifacts: dict[str, str] = field(default_factory=dict)
    eval_summary: dict[str, object] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def eval_summary_from_report(report: dict[str, object]) -> dict[str, object]:
    """Extract the headline delta + verdict from an eval report dict."""
    headline = report.get("headline", {})
    delta = headline.get("delta", {}) if isinstance(headline, dict) else {}
    return {
        "base_mean": headline.get("base_mean") if isinstance(headline, dict) else None,
        "tuned_mean": headline.get("tuned_mean") if isinstance(headline, dict) else None,
        "delta": delta.get("delta"),
        "ci_low": delta.get("ci_low"),
        "ci_high": delta.get("ci_high"),
        "p_value_status": delta.get("p_value_status"),
        "verdict": report.get("verdict"),
    }


def build_record(
    config: Config,
    *,
    data_version: str,
    artifacts: dict[str, str] | None = None,
    eval_summary: dict[str, object] | None = None,
    note: str = "",
) -> RunRecord:
    """Assemble a RunRecord (pure — no I/O beyond reading the git commit)."""
    return RunRecord(
        run_id=ver.run_id(config, data_version),
        created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=ver.git_commit(),
        backend=config.backend,
        base_model=config.model.name,
        data_version=data_version,
        config_fingerprint=ver.config_fingerprint(config),
        artifacts=artifacts or {},
        eval_summary=eval_summary or {},
        note=note,
    )


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> list[RunRecord]:
    """Read all run records; returns [] if the registry doesn't exist yet."""
    p = Path(path)
    if not p.is_file():
        return []
    records: list[RunRecord] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(RunRecord(**json.loads(line)))
    return records


def find_run(run_id: str, path: str | Path = DEFAULT_REGISTRY) -> RunRecord | None:
    """Look up a run by id, or None if not registered."""
    for record in load_registry(path):
        if record.run_id == run_id:
            return record
    return None


def write_registry(records: list[RunRecord], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return out


def register_run(record: RunRecord, path: str | Path = DEFAULT_REGISTRY) -> RunRecord:
    """Upsert a record into the registry by run_id (updates in place)."""
    records = [r for r in load_registry(path) if r.run_id != record.run_id]
    records.append(record)
    write_registry(records, path)
    return record
