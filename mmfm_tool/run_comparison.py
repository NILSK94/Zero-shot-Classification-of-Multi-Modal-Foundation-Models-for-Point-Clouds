from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .evaluation import load_results_csv


PROTOCOL_LABELS: dict[str, str] = {
    "current_paper_protocol": "Current · 33-class prompt",
    "modelnet33_exact_ids": "Legacy · 40-class prompt",
    "modelnet40_legacy_full": "Legacy · full ModelNet40 audit",
}

def run_scope(run_dir: str | Path) -> str:
    config_path = Path(run_dir) / "config.json"
    if not config_path.exists():
        return "current_paper_protocol"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return "current_paper_protocol"
    return str(config.get("scope", "current_paper_protocol") or "current_paper_protocol")


def merge_run_results(
    run_dirs: Iterable[str | Path],
    *,
    keep_duplicate_tasks: bool = False,
) -> pd.DataFrame:
    """Load selected runs into one normalized frame.

    By default exact task IDs are de-duplicated within a protocol. Successful
    rows take precedence over failed rows and the newest row wins ties. This
    lets split or resumed runs be combined without inflating their sample size.
    """

    frames: list[pd.DataFrame] = []
    for value in run_dirs:
        run_dir = Path(value).expanduser().resolve()
        results_path = run_dir / "results.csv"
        if not results_path.exists():
            continue
        frame = load_results_csv(results_path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["source_run"] = run_dir.name
        frame["protocol_scope"] = run_scope(run_dir)
        frame["original_repeat"] = pd.to_numeric(frame.get("repeat"), errors="coerce")
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["task_id"] = combined.get("task_id", "").fillna("").astype(str)
    if keep_duplicate_tasks:
        return combined.reset_index(drop=True)

    with_id = combined[combined["task_id"].str.strip() != ""].copy()
    without_id = combined[combined["task_id"].str.strip() == ""].copy()
    if not with_id.empty:
        with_id["_success_rank"] = with_id.get("status", "").astype(str).eq("ok").astype(int)
        with_id["_timestamp_rank"] = pd.to_datetime(with_id.get("timestamp"), errors="coerce", utc=True)
        with_id = with_id.sort_values(
            ["protocol_scope", "task_id", "_success_rank", "_timestamp_rank", "source_run"],
            na_position="first",
        ).drop_duplicates(["protocol_scope", "task_id"], keep="last")
        with_id = with_id.drop(columns=["_success_rank", "_timestamp_rank"])
    return pd.concat([with_id, without_id], ignore_index=True, sort=False).reset_index(drop=True)


def summarize_merged_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    data = results.copy()
    data["view_count"] = pd.to_numeric(data.get("view_count"), errors="coerce")
    data["is_success"] = data.get("status", "").astype(str).eq("ok")
    true_label = data.get("true_label", "").fillna("").astype(str).str.casefold()
    pred_label = data.get("pred_label", "").fillna("").astype(str).str.casefold()
    data["is_correct"] = data["is_success"] & true_label.eq(pred_label)
    data["latency_numeric"] = pd.to_numeric(data.get("latency_s"), errors="coerce")

    groups = [
        "protocol_scope",
        "provider",
        "model",
        "dataset",
        "condition",
        "view_count",
    ]
    summary = (
        data.groupby(groups, dropna=False, sort=True)
        .agg(
            n_runs=("source_run", "nunique"),
            n_planned=("sample_id", "size"),
            n_objects=("sample_id", "nunique"),
            n_classes=("true_label", "nunique"),
            n_success=("is_success", "sum"),
            n_correct=("is_correct", "sum"),
            mean_latency_s=("latency_numeric", "mean"),
        )
        .reset_index()
    )
    summary["coverage"] = summary["n_success"] / summary["n_planned"].replace(0, pd.NA)
    summary["end_to_end_accuracy"] = summary["n_correct"] / summary["n_planned"].replace(0, pd.NA)
    summary["accuracy_valid_only"] = summary["n_correct"] / summary["n_success"].replace(0, pd.NA)
    summary["protocol"] = summary["protocol_scope"].map(PROTOCOL_LABELS).fillna(summary["protocol_scope"])
    summary["model_label"] = summary["provider"].astype(str) + " / " + summary["model"].astype(str)
    summary["comparison_series"] = summary["protocol"] + " · " + summary["model_label"]
    return summary


def modelnet_view_table(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    if summary.empty or metric not in summary:
        return pd.DataFrame()
    data = summary[summary["dataset"].astype(str).str.casefold() == "modelnet"].copy()
    if data.empty:
        return pd.DataFrame()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    table = data.pivot_table(
        index=["model"],
        columns="view_count",
        values=metric,
        aggfunc="mean",
    ).reindex(columns=[1, 2, 3, 4])
    table.columns = [f"{view} view" if view == 1 else f"{view} views" for view in table.columns]
    table = table.reset_index().rename(columns={"model": "Model"})
    valid_data = data[pd.to_numeric(data.get("n_success"), errors="coerce").fillna(0).gt(0)]
    available = (
        valid_data.groupby(["model"], dropna=False)["view_count"]
        .apply(lambda values: ", ".join(str(int(value)) for value in sorted(set(values.dropna()))))
        .rename("Views with valid responses")
        .reset_index()
        .rename(columns={"model": "Model"})
    )
    table = table.merge(available, on=["Model"], how="left")
    table["Views with valid responses"] = table["Views with valid responses"].fillna("None")
    return table
