from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .evaluation import evaluate_results
from .runner import RESULT_FIELDS

LEGACY_REQUIRED_COLUMNS = {
    "sample_id",
    "ground_truth",
    "predicted_label",
    "model_provider",
    "model_name",
    "prompt_name",
    "num_views",
    "run_index",
    "runtime_seconds",
    "raw_response",
}


class LegacyResultError(ValueError):
    pass


@dataclass(frozen=True)
class LegacyResultBundle:
    results: pd.DataFrame
    profile: pd.DataFrame
    labels: tuple[str, ...]


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _task_hash(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _normalize_file(path: Path, frame: pd.DataFrame, labels: tuple[str, ...]) -> pd.DataFrame:
    missing_columns = sorted(LEGACY_REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise LegacyResultError(f"{path.name}: Required columns are missing: {', '.join(missing_columns)}")

    allowed = {label.casefold(): label for label in labels}
    timestamp = _file_timestamp(path)
    rows: list[dict[str, object]] = []
    for source_index, source in frame.fillna("").iterrows():
        true_label = _text(source.get("ground_truth")).casefold()
        legacy_sample_id = _text(source.get("sample_id")).replace("\\", "/")
        sample_id = legacy_sample_id if "/" in legacy_sample_id else f"{true_label}/{legacy_sample_id}"
        predicted_value = _text(source.get("predicted_label"))
        raw_response = _text(source.get("raw_response"))
        candidate = predicted_value or raw_response
        normalized_prediction = allowed.get(candidate.casefold(), "")
        status = "ok" if normalized_prediction else "error"
        if status == "ok":
            error = ""
        elif not candidate:
            error = 'The historical file contains no model response.'
        else:
            error = 'The historical response is not an exact class name from the historical ModelNet40 label set.'

        view_count = int(pd.to_numeric(source.get("num_views"), errors="coerce"))
        repeat = int(pd.to_numeric(source.get("run_index"), errors="coerce")) + 1
        provider = _text(source.get("model_provider")).casefold() or "legacy"
        model = _text(source.get("model_name")) or path.stem
        prompt_name = _text(source.get("prompt_name")) or "unknown"
        usage = {
            "input_tokens": pd.to_numeric(source.get("input_tokens"), errors="coerce"),
            "output_tokens": pd.to_numeric(source.get("output_tokens"), errors="coerce"),
            "estimated_cost_usd": pd.to_numeric(source.get("estimated_cost_usd"), errors="coerce"),
        }
        usage = {
            key: (None if pd.isna(value) else float(value))
            for key, value in usage.items()
        }
        rows.append(
            {
                "task_id": _task_hash("legacy", path.name, sample_id, view_count, repeat),
                "timestamp": timestamp,
                "dataset": "modelnet",
                "sample_id": sample_id,
                "true_label": true_label,
                "provider": provider,
                "model": model,
                "model_version": model,
                "condition": f"views_{view_count}",
                "view_count": view_count,
                "repeat": repeat,
                "status": status,
                "pred_label": normalized_prediction,
                "correct": normalized_prediction == true_label if status == "ok" else False,
                "latency_s": pd.to_numeric(source.get("runtime_seconds"), errors="coerce"),
                "error": error,
                "usage_json": json.dumps(usage, ensure_ascii=False, sort_keys=True),
                "geometry_json": "{}",
                "image_paths_json": "[]",
                "prompt_version": f"legacy:{prompt_name}",
                "prompt_hash": "",
                "prompt_text": "",
                "raw_response": raw_response,
                "legacy_source_file": path.name,
                "legacy_source_row": int(source_index) + 2,
                "evaluation_scope": "modelnet40_legacy_full",
                "historical_label_space": len(labels),
            }
        )
    return pd.DataFrame(rows)


def load_legacy_results(root: str | Path) -> LegacyResultBundle:
    source_root = Path(root).expanduser().resolve()
    if not source_root.is_dir():
        raise LegacyResultError(f"Historical results folder not found: {source_root}")
    source_frames: list[tuple[Path, pd.DataFrame]] = []
    labels: set[str] = set()
    for path in sorted(source_root.glob("*.csv")):
        frame = pd.read_csv(path).fillna("")
        if "ground_truth" in frame:
            labels.update(frame["ground_truth"].astype(str).str.strip().str.casefold())
        source_frames.append((path, frame))
    if not source_frames:
        raise LegacyResultError(f"No CSV files under {source_root} found.")

    label_tuple = tuple(sorted(label for label in labels if label))
    normalized_frames = [_normalize_file(path, frame, label_tuple) for path, frame in source_frames]
    results = pd.concat(normalized_frames, ignore_index=True)
    expected_full_rows = int(max(frame["sample_id"].nunique() for frame in normalized_frames))
    profile_rows: list[dict[str, object]] = []
    for path, normalized in zip((item[0] for item in source_frames), normalized_frames):
        success = normalized[normalized["status"] == "ok"]
        correct = int(success["correct"].astype(bool).sum())
        profile_rows.append(
            {
                "source_file": path.name,
                "provider": normalized["provider"].iloc[0],
                "model": normalized["model"].iloc[0],
                "view_count": int(normalized["view_count"].iloc[0]),
                "rows_present": len(normalized),
                "expected_modelnet40_rows": expected_full_rows,
                "row_completeness": len(normalized) / expected_full_rows if expected_full_rows else 0.0,
                "ground_truth_classes": int(normalized["true_label"].nunique()),
                "valid_responses": len(success),
                "coverage": len(success) / len(normalized) if len(normalized) else 0.0,
                "correct_responses": correct,
                "accuracy_valid_only": correct / len(success) if len(success) else float("nan"),
                "end_to_end_accuracy": correct / len(normalized) if len(normalized) else 0.0,
                "file_state": 'complete' if len(normalized) == expected_full_rows else 'partial run',
            }
        )
    profile = pd.DataFrame(profile_rows).sort_values(["provider", "model", "view_count"]).reset_index(drop=True)
    return LegacyResultBundle(results=results, profile=profile, labels=label_tuple)


def _align_to_current_modelnet(
    full_results: pd.DataFrame,
    modelnet_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = (
        modelnet_manifest[["sample_id", "label"]]
        .drop_duplicates()
        .rename(columns={"label": "expected_label"})
        .sort_values(["expected_label", "sample_id"])
        .reset_index(drop=True)
    )
    if expected.empty:
        raise LegacyResultError('The current ModelNet manifest contains no comparable test objects.')

    aligned_frames: list[pd.DataFrame] = []
    for source_file, group in full_results.groupby("legacy_source_file", sort=True):
        group = group.sort_values("legacy_source_row").drop_duplicates("sample_id", keep="last")
        metadata = group.iloc[0]
        aligned = expected.merge(group, on="sample_id", how="left", indicator=True)
        missing = aligned["_merge"] == "left_only"
        aligned["true_label"] = aligned["expected_label"]
        for column in (
            "provider",
            "model",
            "model_version",
            "condition",
            "view_count",
            "repeat",
            "timestamp",
            "prompt_version",
            "historical_label_space",
        ):
            aligned[column] = aligned[column].fillna(metadata[column])
        aligned["legacy_source_file"] = source_file
        aligned.loc[missing, "legacy_source_row"] = float("nan")
        aligned.loc[missing, "status"] = "error"
        aligned.loc[missing, "pred_label"] = ""
        aligned.loc[missing, "correct"] = False
        aligned.loc[missing, "latency_s"] = float("nan")
        aligned.loc[missing, "error"] = 'Partial historical run: the result row for this comparison object is missing.'
        for column, default in (
            ("usage_json", "{}"),
            ("geometry_json", "{}"),
            ("image_paths_json", "[]"),
            ("prompt_hash", ""),
            ("prompt_text", ""),
            ("raw_response", ""),
        ):
            aligned[column] = aligned[column].fillna(default)
        aligned["dataset"] = "modelnet"
        aligned["evaluation_scope"] = "modelnet33_exact_ids"
        aligned["task_id"] = aligned.apply(
            lambda row: _task_hash(
                "legacy-common33",
                source_file,
                row["sample_id"],
                int(row["view_count"]),
                int(row["repeat"]),
            ),
            axis=1,
        )
        aligned = aligned.drop(columns=["expected_label", "_merge"])
        aligned_frames.append(aligned)

    comparable = pd.concat(aligned_frames, ignore_index=True)
    # The current ground truth is authoritative for the exact common cohort.
    comparable["correct"] = (
        (comparable["status"] == "ok")
        & (comparable["true_label"].astype(str).str.casefold() == comparable["pred_label"].astype(str).str.casefold())
    )
    return comparable, expected.rename(columns={"expected_label": "label"})


def _write_evaluation_run(
    run_dir: Path,
    results: pd.DataFrame,
    samples: pd.DataFrame,
    profile: pd.DataFrame,
    config: dict[str, object],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    ordered_columns = [*RESULT_FIELDS, *[column for column in results.columns if column not in RESULT_FIELDS]]
    results[ordered_columns].to_csv(run_dir / "results.csv", index=False, encoding="utf-8")
    samples.to_csv(run_dir / "samples.csv", index=False, encoding="utf-8")
    profile.to_csv(run_dir / "legacy_source_profile.csv", index=False, encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    evaluate_results(results, run_dir)


def import_legacy_results(
    old_results_root: str | Path,
    modelnet_manifest: pd.DataFrame,
    runs_root: str | Path,
) -> dict[str, Path]:
    bundle = load_legacy_results(old_results_root)
    comparable, comparable_samples = _align_to_current_modelnet(bundle.results, modelnet_manifest)
    destination = Path(runs_root).expanduser().resolve()
    full_dir = destination / "legacy_old_results_modelnet40"
    comparable_dir = destination / "legacy_old_results_modelnet33_comparable"
    source_root = Path(old_results_root).expanduser().resolve()

    full_samples = (
        bundle.results[["sample_id", "true_label"]]
        .drop_duplicates()
        .rename(columns={"true_label": "label"})
        .sort_values(["label", "sample_id"])
    )
    common_config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source_root),
        "source_files": int(bundle.profile["source_file"].nunique()),
        "dataset": "modelnet",
        "scope": "modelnet33_exact_ids",
        "selected_objects": int(comparable_samples["sample_id"].nunique()),
        "selected_classes": int(comparable_samples["label"].nunique()),
        "historical_prompt_label_space": len(bundle.labels),
        "note": (
            "Rows are aligned to the exact current Modelnet_TEST_1 object IDs. Missing rows from partial legacy files "
            "are explicit errors. Historical prompts may still have offered the original 40-class label space."
        ),
    }
    full_config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source_root),
        "source_files": int(bundle.profile["source_file"].nunique()),
        "dataset": "modelnet",
        "scope": "modelnet40_legacy_full",
        "selected_objects": int(full_samples["sample_id"].nunique()),
        "selected_classes": int(full_samples["label"].nunique()),
        "note": "Unmodified historical ModelNet40 cohort retained for auditability.",
    }
    _write_evaluation_run(full_dir, bundle.results, full_samples, bundle.profile, full_config)
    _write_evaluation_run(comparable_dir, comparable, comparable_samples, bundle.profile, common_config)
    return {
        "full40": full_dir,
        "comparable33": comparable_dir,
        "profile": comparable_dir / "legacy_source_profile.csv",
    }
