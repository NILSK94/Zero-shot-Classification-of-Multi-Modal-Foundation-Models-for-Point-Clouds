from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

GROUP_COLUMNS = ["provider", "model", "dataset", "condition", "view_count", "repeat"]


def _wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    proportion = correct / total
    denominator = 1.0 + (z * z / total)
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt((proportion * (1.0 - proportion) / total) + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _mcnemar_exact(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    try:
        from scipy.stats import binomtest

        return float(binomtest(min(gains, losses), discordant, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        # Continuity-corrected chi-square approximation as a dependency-safe fallback.
        statistic = (abs(gains - losses) - 1.0) ** 2 / discordant
        return float(math.erfc(math.sqrt(max(statistic, 0.0) / 2.0)))


def _latest_attempts(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    if "timestamp" in frame:
        frame = frame.sort_values("timestamp")
    if "task_id" in frame:
        frame = frame.drop_duplicates("task_id", keep="last")
    return frame.reset_index(drop=True)


def evaluate_results(results: pd.DataFrame, output_dir: str | Path) -> dict[str, pd.DataFrame]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    latest = _latest_attempts(results)
    if latest.empty:
        empty = pd.DataFrame()
        return {"summary": empty, "aggregate": empty, "per_class": empty, "confusion": empty, "paired": empty}

    for column in ("view_count", "repeat"):
        if column in latest:
            latest[column] = pd.to_numeric(latest[column], errors="coerce")

    summary_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []

    for group_key, planned_group in latest.groupby(GROUP_COLUMNS, dropna=False, sort=True):
        identity = dict(zip(GROUP_COLUMNS, group_key))
        valid = planned_group[planned_group["status"] == "ok"].copy()
        total = len(planned_group)
        success = len(valid)
        if success == 0:
            ci_low, ci_high = _wilson_interval(0, total)
            summary_rows.append(
                {
                    **identity,
                    "n_planned": total,
                    "n_success": 0,
                    "coverage": 0.0,
                    "end_to_end_accuracy": 0.0,
                    "end_to_end_accuracy_ci95_low": ci_low,
                    "end_to_end_accuracy_ci95_high": ci_high,
                }
            )
            continue

        truth = valid["true_label"].astype(str).str.lower()
        prediction = valid["pred_label"].astype(str).str.lower()
        labels = sorted(set(truth) | set(prediction))
        correct = int((truth == prediction).sum())
        ci_low, ci_high = _wilson_interval(correct, success)
        end_to_end_ci_low, end_to_end_ci_high = _wilson_interval(correct, total)
        summary_rows.append(
            {
                **identity,
                "n_planned": total,
                "n_success": success,
                "coverage": success / total if total else 0.0,
                "accuracy": accuracy_score(truth, prediction),
                "accuracy_ci95_low": ci_low,
                "accuracy_ci95_high": ci_high,
                "end_to_end_accuracy": correct / total if total else 0.0,
                "end_to_end_accuracy_ci95_low": end_to_end_ci_low,
                "end_to_end_accuracy_ci95_high": end_to_end_ci_high,
                "balanced_accuracy": balanced_accuracy_score(truth, prediction),
                "macro_precision": precision_score(truth, prediction, labels=labels, average="macro", zero_division=0),
                "macro_recall": recall_score(truth, prediction, labels=labels, average="macro", zero_division=0),
                "macro_f1": f1_score(truth, prediction, labels=labels, average="macro", zero_division=0),
                "weighted_f1": f1_score(truth, prediction, labels=labels, average="weighted", zero_division=0),
                "mean_latency_s": pd.to_numeric(valid["latency_s"], errors="coerce").mean(),
            }
        )

        precision, recall, f1, support = precision_recall_fscore_support(
            truth,
            prediction,
            labels=labels,
            zero_division=0,
        )
        for label, p_value, r_value, f_value, count in zip(labels, precision, recall, f1, support):
            class_rows.append(
                {
                    **identity,
                    "label": label,
                    "precision": p_value,
                    "recall": r_value,
                    "f1": f_value,
                    "support": int(count),
                }
            )

        matrix = confusion_matrix(truth, prediction, labels=labels)
        for true_index, true_label in enumerate(labels):
            for predicted_index, predicted_label in enumerate(labels):
                confusion_rows.append(
                    {
                        **identity,
                        "true_label": true_label,
                        "pred_label": predicted_label,
                        "count": int(matrix[true_index, predicted_index]),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    per_class = pd.DataFrame(class_rows)
    confusion = pd.DataFrame(confusion_rows)

    numeric_metrics = [
        "coverage",
        "accuracy",
        "end_to_end_accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "mean_latency_s",
    ]
    aggregate_rows: list[dict[str, object]] = []
    aggregate_groups = ["provider", "model", "dataset", "condition", "view_count"]
    if not summary.empty:
        for group_key, group in summary.groupby(aggregate_groups, dropna=False, sort=True):
            row = dict(zip(aggregate_groups, group_key))
            row["repeats"] = int(group["repeat"].nunique())
            row["n_success_total"] = int(pd.to_numeric(group["n_success"], errors="coerce").sum())
            for metric in numeric_metrics:
                if metric not in group:
                    row[f"{metric}_mean"] = float("nan")
                    row[f"{metric}_std"] = float("nan")
                    continue
                values = pd.to_numeric(group[metric], errors="coerce")
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=1) if values.notna().sum() > 1 else 0.0
            aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)

    paired = paired_comparisons(latest)
    outputs = {
        "summary": summary,
        "aggregate": aggregate,
        "per_class": per_class,
        "confusion": confusion,
        "paired": paired,
    }
    filenames = {
        "summary": "summary_by_repeat.csv",
        "aggregate": "summary_aggregate.csv",
        "per_class": "per_class_metrics.csv",
        "confusion": "confusion_matrix_long.csv",
        "paired": "paired_comparisons.csv",
    }
    for name, frame in outputs.items():
        frame.to_csv(destination / filenames[name], index=False, encoding="utf-8")
    return outputs


def _pair_two_conditions(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
    identity: dict[str, object],
) -> dict[str, object] | None:
    left_values = left[["sample_id", "true_label", "pred_label", "status"]].rename(
        columns={"true_label": "truth_left", "pred_label": "pred_left", "status": "status_left"}
    )
    right_values = right[["sample_id", "true_label", "pred_label", "status"]].rename(
        columns={"true_label": "truth_right", "pred_label": "pred_right", "status": "status_right"}
    )
    merged = left_values.merge(right_values, on="sample_id", how="inner")
    if merged.empty:
        return None
    consistent = merged[merged["truth_left"] == merged["truth_right"]].copy()
    if consistent.empty:
        return None
    left_valid = consistent["status_left"].astype(str) == "ok"
    right_valid = consistent["status_right"].astype(str) == "ok"
    left_correct = left_valid & (
        consistent["truth_left"].astype(str).str.lower() == consistent["pred_left"].astype(str).str.lower()
    )
    right_correct = right_valid & (
        consistent["truth_right"].astype(str).str.lower() == consistent["pred_right"].astype(str).str.lower()
    )
    gains = int((~left_correct & right_correct).sum())
    losses = int((left_correct & ~right_correct).sum())
    return {
        **identity,
        "left_condition": left_name,
        "right_condition": right_name,
        "n_paired": len(consistent),
        "left_coverage": float(left_valid.mean()),
        "right_coverage": float(right_valid.mean()),
        "n_both_success": int((left_valid & right_valid).sum()),
        "left_accuracy": float(left_correct.mean()),
        "right_accuracy": float(right_correct.mean()),
        "delta_accuracy": float(right_correct.mean() - left_correct.mean()),
        "right_gains": gains,
        "right_losses": losses,
        "mcnemar_exact_p": _mcnemar_exact(gains, losses),
    }


def paired_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    planned = results.copy()
    rows: list[dict[str, object]] = []
    base_groups = ["provider", "model", "repeat"]
    for group_key, model_group in planned.groupby(base_groups, dropna=False, sort=True):
        identity = dict(zip(base_groups, group_key))
        datasets = set(model_group["dataset"].astype(str))
        if "s3dis" in datasets:
            data = model_group[model_group["dataset"] == "s3dis"]
            left = data[data["condition"] == "rgb"]
            right = data[data["condition"] == "rgb_aabb"]
            comparison = _pair_two_conditions(left, right, "rgb", "rgb_aabb", {**identity, "dataset": "s3dis"})
            if comparison:
                rows.append(comparison)
        if "modelnet" in datasets:
            data = model_group[model_group["dataset"] == "modelnet"]
            left = data[pd.to_numeric(data["view_count"], errors="coerce") == 1]
            for view_count in sorted(set(pd.to_numeric(data["view_count"], errors="coerce").dropna().astype(int)) - {1}):
                right = data[pd.to_numeric(data["view_count"], errors="coerce") == view_count]
                comparison = _pair_two_conditions(
                    left,
                    right,
                    "views_1",
                    f"views_{view_count}",
                    {**identity, "dataset": "modelnet"},
                )
                if comparison:
                    rows.append(comparison)
    columns = [
        "provider",
        "model",
        "repeat",
        "dataset",
        "left_condition",
        "right_condition",
        "n_paired",
        "left_coverage",
        "right_coverage",
        "n_both_success",
        "left_accuracy",
        "right_accuracy",
        "delta_accuracy",
        "right_gains",
        "right_losses",
        "mcnemar_exact_p",
    ]
    return pd.DataFrame(rows, columns=columns)


def load_results_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(csv_path).fillna("")
