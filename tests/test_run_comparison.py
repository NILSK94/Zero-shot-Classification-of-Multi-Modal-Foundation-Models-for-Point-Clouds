from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mmfm_tool.run_comparison import (
    merge_run_results,
    modelnet_view_table,
    summarize_merged_results,
)


def _write_run(path: Path, scope: str, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"scope": scope}), encoding="utf-8")
    pd.DataFrame(rows).to_csv(path / "results.csv", index=False)


def _row(task_id: str, view_count: int, status: str, pred_label: str, timestamp: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "timestamp": timestamp,
        "dataset": "modelnet",
        "sample_id": "chair/chair_0001",
        "true_label": "chair",
        "provider": "mock",
        "model": "mock-test",
        "condition": f"views_{view_count}",
        "view_count": view_count,
        "repeat": 1,
        "status": status,
        "pred_label": pred_label,
        "latency_s": 1.0,
    }


def test_merge_selected_runs_deduplicates_resumed_tasks_and_prefers_success(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_run(
        first,
        "current_paper_protocol",
        [_row("same-task", 1, "error", "", "2026-01-01T00:00:00Z")],
    )
    _write_run(
        second,
        "current_paper_protocol",
        [
            _row("same-task", 1, "ok", "chair", "2026-01-02T00:00:00Z"),
            _row("view-two", 2, "ok", "chair", "2026-01-02T00:00:01Z"),
        ],
    )

    merged = merge_run_results([first, second])

    assert len(merged) == 2
    assert merged.loc[merged["task_id"] == "same-task", "status"].item() == "ok"
    assert set(merged["source_run"]) == {"second"}


def test_old_and_new_summaries_keep_fixed_one_to_four_view_columns(tmp_path: Path) -> None:
    current = tmp_path / "current"
    legacy = tmp_path / "legacy"
    _write_run(
        current,
        "current_paper_protocol",
        [
            _row("new-one", 1, "ok", "chair", "2026-01-01T00:00:00Z"),
            _row("new-four", 4, "ok", "chair", "2026-01-01T00:00:01Z"),
        ],
    )
    _write_run(
        legacy,
        "modelnet33_exact_ids",
        [_row("old-two", 2, "ok", "table", "2025-01-01T00:00:00Z")],
    )

    summary = summarize_merged_results(merge_run_results([current, legacy]))
    table = modelnet_view_table(summary, "end_to_end_accuracy")

    assert set(summary["protocol"]) == {"Current · 33-class prompt", "Legacy · 40-class prompt"}
    assert [column for column in table if "Ansicht" in column] == [
    ]
    assert list(table.columns) == [
        "Model",
        "1 view",
        "2 views",
        "3 views",
        "4 views",
        "Views with valid responses",
    ]
    current_row = table.iloc[0]
    assert current_row["Views with valid responses"] == "1, 2, 4"
    assert pd.isna(current_row["3 views"])
