from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .evaluation import GROUP_COLUMNS, evaluate_results, load_results_csv


@dataclass(frozen=True)
class PaperExport:
    output_dir: Path
    zip_path: Path
    files: dict[str, Path]


def _scope_from_config(run_dir: Path) -> str:
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            scope = str(config.get("scope", "")).strip()
            if scope:
                return scope
        except (OSError, ValueError, TypeError):
            pass
    return "current_paper_protocol"


def _safe_slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_") or "scope"


def _load_comparable_runs(runs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []
    for run_dir in sorted(path for path in runs_root.glob("*") if path.is_dir()):
        results_path = run_dir / "results.csv"
        if not results_path.exists():
            continue
        scope = _scope_from_config(run_dir)
        results = load_results_csv(results_path)
        inventory_rows.append(
            {
                "run": run_dir.name,
                "scope": scope,
                "rows": len(results),
                "objects": int(results["sample_id"].nunique()) if "sample_id" in results else 0,
                "models": int(results[["provider", "model"]].drop_duplicates().shape[0])
                if {"provider", "model"}.issubset(results.columns)
                else 0,
                "included_in_comparable_export": scope != "modelnet40_legacy_full",
            }
        )
        if results.empty or scope == "modelnet40_legacy_full":
            continue
        results["source_run"] = run_dir.name
        results["protocol_scope"] = scope
        frames.append(results)
    if not frames:
        return pd.DataFrame(), pd.DataFrame(inventory_rows)
    combined = pd.concat(frames, ignore_index=True)
    if "timestamp" in combined:
        combined = combined.sort_values("timestamp")
    if "task_id" in combined:
        combined = combined.drop_duplicates("task_id", keep="last")
    return combined.reset_index(drop=True), pd.DataFrame(inventory_rows)


def _evaluate_by_scope(results: pd.DataFrame, metrics_root: Path) -> dict[str, pd.DataFrame]:
    collected: dict[str, list[pd.DataFrame]] = {
        "summary": [],
        "aggregate": [],
        "per_class": [],
        "confusion": [],
        "paired": [],
    }
    for scope, group in results.groupby("protocol_scope", sort=True):
        scope_dir = metrics_root / _safe_slug(str(scope))
        evaluated = evaluate_results(group, scope_dir)
        for name, frame in evaluated.items():
            if frame.empty:
                continue
            enriched = frame.copy()
            enriched.insert(0, "protocol_scope", scope)
            collected[name].append(enriched)
    return {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in collected.items()
    }


def _cohort_stats(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeat_groups = ["protocol_scope", *GROUP_COLUMNS]
    repeat_stats = (
        results.groupby(repeat_groups, dropna=False, sort=True)
        .agg(
            n_objects=("sample_id", "nunique"),
            n_classes=("true_label", "nunique"),
        )
        .reset_index()
    )
    aggregate_groups = ["protocol_scope", "provider", "model", "dataset", "condition", "view_count"]
    aggregate_stats = (
        repeat_stats.groupby(aggregate_groups, dropna=False, sort=True)
        .agg(
            n_objects_min=("n_objects", "min"),
            n_objects_max=("n_objects", "max"),
            n_classes_min=("n_classes", "min"),
            n_classes_max=("n_classes", "max"),
        )
        .reset_index()
    )
    return repeat_stats, aggregate_stats


def _paper_status(row: pd.Series) -> str:
    if str(row.get("provider", "")).casefold() == "mock":
        return "Nur Pipeline-Test"
    if str(row.get("dataset", "")).casefold() != "modelnet":
        return "Datensatzspezifische Auswertung"
    exact_cohort = (
        int(row.get("n_objects_min", 0)) == 1860
        and int(row.get("n_objects_max", 0)) == 1860
        and int(row.get("n_classes_min", 0)) == 33
        and int(row.get("n_classes_max", 0)) == 33
    )
    if not exact_cohort:
        return "Teillauf – nicht direkt vergleichbar"
    coverage = pd.to_numeric(row.get("coverage_mean"), errors="coerce")
    if pd.notna(coverage) and float(coverage) >= 0.95:
        if str(row.get("protocol_scope", "")) == "modelnet33_exact_ids":
            return "Haupttabelle – historischer 40-Label-Prompt"
        return "Haupttabelle geeignet"
    return "Exakter Datensatz, aber unvollständige Modellantworten"


def _s3dis_area_summary(results: pd.DataFrame) -> pd.DataFrame:
    data = results[results["dataset"].astype(str).str.casefold() == "s3dis"].copy()
    if data.empty:
        return pd.DataFrame()
    data["area"] = data["sample_id"].astype(str).str.split("/").str[0]
    data["is_success"] = data["status"].astype(str) == "ok"
    data["is_correct"] = data["is_success"] & (
        data["true_label"].astype(str).str.casefold() == data["pred_label"].astype(str).str.casefold()
    )
    groups = [
        "protocol_scope",
        "provider",
        "model",
        "condition",
        "view_count",
        "repeat",
        "area",
    ]
    summary = (
        data.groupby(groups, dropna=False, sort=True)
        .agg(
            n_planned=("sample_id", "size"),
            n_objects=("sample_id", "nunique"),
            n_classes=("true_label", "nunique"),
            n_success=("is_success", "sum"),
            n_correct=("is_correct", "sum"),
            mean_latency_s=("latency_s", lambda values: pd.to_numeric(values, errors="coerce").mean()),
        )
        .reset_index()
    )
    summary["coverage"] = summary["n_success"] / summary["n_planned"]
    summary["end_to_end_accuracy"] = summary["n_correct"] / summary["n_planned"]
    summary["accuracy_valid_only"] = summary["n_correct"] / summary["n_success"].replace(0, pd.NA)
    return summary


def build_paper_export(runs_root: str | Path, output_root: str | Path) -> PaperExport:
    runs = Path(runs_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    combined, inventory = _load_comparable_runs(runs)
    if combined.empty:
        raise ValueError(f"Keine auswertbaren Ergebnisläufe unter {runs} gefunden.")

    metrics_root = output / "_metrics_by_protocol"
    evaluated = _evaluate_by_scope(combined, metrics_root)
    repeat_stats, aggregate_stats = _cohort_stats(combined)
    summary = evaluated["summary"].merge(
        repeat_stats,
        on=["protocol_scope", *GROUP_COLUMNS],
        how="left",
    )
    aggregate = evaluated["aggregate"].merge(
        aggregate_stats,
        on=["protocol_scope", "provider", "model", "dataset", "condition", "view_count"],
        how="left",
    )
    aggregate["paper_status"] = aggregate.apply(_paper_status, axis=1)
    main_mask = aggregate["paper_status"].astype(str).str.startswith("Haupttabelle")
    main = aggregate[main_mask].copy().sort_values(["provider", "model", "view_count"])
    incomplete = aggregate[~main_mask].copy()
    s3dis_areas = _s3dis_area_summary(combined)
    protocol_labels = {
        "current_paper_protocol": "Current paper protocol (33-label prompt)",
        "modelnet33_exact_ids": "Legacy 40-label prompt on exact common cohort",
    }
    manuscript_columns = [
        "protocol_scope",
        "provider",
        "model",
        "view_count",
        "repeats",
        "n_objects_min",
        "n_classes_min",
        "coverage_mean",
        "end_to_end_accuracy_mean",
        "accuracy_mean",
        "macro_f1_mean",
        "balanced_accuracy_mean",
        "mean_latency_s_mean",
        "paper_status",
    ]
    manuscript = main[[column for column in manuscript_columns if column in main]].copy()
    manuscript["protocol_scope"] = manuscript["protocol_scope"].map(protocol_labels).fillna(
        manuscript["protocol_scope"]
    )
    manuscript = manuscript.rename(
        columns={
            "protocol_scope": "Protocol",
            "provider": "Provider",
            "model": "Model",
            "view_count": "Views",
            "repeats": "Repeats",
            "n_objects_min": "Objects",
            "n_classes_min": "Classes",
            "coverage_mean": "Coverage",
            "end_to_end_accuracy_mean": "End-to-end Accuracy",
            "accuracy_mean": "Valid-only Accuracy",
            "macro_f1_mean": "Macro F1 (valid only)",
            "balanced_accuracy_mean": "Balanced Accuracy (valid only)",
            "mean_latency_s_mean": "Mean Runtime (s)",
            "paper_status": "Paper Status",
        }
    )

    legacy_quality_path = runs / "legacy_old_results_modelnet33_comparable" / "legacy_source_profile.csv"
    legacy_quality = pd.read_csv(legacy_quality_path).fillna("") if legacy_quality_path.exists() else pd.DataFrame()
    legacy_full_summary_path = runs / "legacy_old_results_modelnet40" / "summary_by_repeat.csv"
    legacy_full_summary = (
        pd.read_csv(legacy_full_summary_path).fillna("") if legacy_full_summary_path.exists() else pd.DataFrame()
    )

    notes = pd.DataFrame(
        [
            {
                "topic": "Main comparison cohort",
                "statement": "ModelNet main tables use exactly 1,860 object IDs and 33 ground-truth classes from Modelnet_TEST_1.",
            },
            {
                "topic": "Historical label space",
                "statement": "Legacy rows are aligned to the common 33-class cohort, but their historical prompts may still have offered 40 labels; this is retained as a protocol difference.",
            },
            {
                "topic": "Coverage",
                "statement": "Coverage is the fraction of planned objects with one exact allowed class-name response.",
            },
            {
                "topic": "End-to-end accuracy",
                "statement": "End-to-end accuracy divides correct predictions by all planned objects and counts missing or invalid responses as incorrect.",
            },
            {
                "topic": "Valid-only accuracy",
                "statement": "Accuracy reports performance only among valid responses and must always be read together with coverage.",
            },
            {
                "topic": "Main-table eligibility",
                "statement": "A model/view enters the recommended main table only with the exact 1,860/33 cohort, at least 95% mean coverage, and a non-mock provider.",
            },
        ]
    )

    tables = {
        "combined_normalized_results": combined,
        "paper_summary_by_repeat": summary,
        "paper_model_view_summary": aggregate,
        "paper_main_comparison": main,
        "paper_manuscript_table": manuscript,
        "paper_incomplete_runs": incomplete,
        "paper_per_class_metrics": evaluated["per_class"],
        "paper_paired_view_comparisons": evaluated["paired"],
        "paper_confusion_matrix_long": evaluated["confusion"],
        "paper_s3dis_area_summary": s3dis_areas,
        "paper_legacy_data_quality": legacy_quality,
        "legacy_modelnet40_audit_summary": legacy_full_summary,
        "run_inventory": inventory,
        "method_notes": notes,
    }
    files: dict[str, Path] = {}
    for name, frame in tables.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        files[name] = path

    zip_path = output / "MMFM_Paper_Results_CSV.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, arcname=path.name)
    return PaperExport(output_dir=output, zip_path=zip_path, files=files)
