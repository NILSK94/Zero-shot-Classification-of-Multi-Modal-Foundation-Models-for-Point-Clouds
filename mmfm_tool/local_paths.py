"""Resolve local data locations without committing machine-specific paths."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping


def data_paths(project_dir: Path, environment: Mapping[str, str] | None = None) -> dict[str, Path]:
    project_dir = Path(project_dir).resolve()
    environment = os.environ if environment is None else environment
    try:
        local = json.loads((project_dir / "data_paths.local.json").read_text(encoding="utf-8"))
        if not isinstance(local, dict):
            local = {}
    except (OSError, ValueError):
        local = {}
    workspace = project_dir.parent
    active = workspace / "Multimodal-point-cloud-classifier-main"
    definitions = {
        "modelnet": ("MODELNET_PATH", [project_dir / "data/modelnet", workspace / "Modelnet_TEST_1"]),
        "s3dis": ("S3DIS_PATH", [project_dir / "data/s3dis", workspace / "S3DIS_TEST"]),
        "prepared": ("S3DIS_PREPARED_PATH", [project_dir / "artifacts/s3dis_prepared", active / "artifacts/s3dis_prepared"]),
        "runs": ("RUNS_PATH", [project_dir / "artifacts/runs", active / "artifacts/runs"]),
        "legacy_results": ("LEGACY_RESULTS_PATH", [project_dir / "data/legacy_results", workspace / "Old_results"]),
        "historical_csv": ("HISTORICAL_RESULTS_PATH", [workspace / "Multimodal_old/Multimodal-point-cloud-classifier-main/data_results_csv", project_dir / "data_results_csv", active / "data_results_csv"]),
        "paper_export": ("PAPER_EXPORT_PATH", [project_dir / "artifacts/paper_exports/current", active / "artifacts/paper_exports/current"]),
    }
    result = {}
    for name, (variable, candidates) in definitions.items():
        explicit = environment.get(variable, "").strip()
        if explicit:
            result[name] = Path(explicit).expanduser()
            continue
        configured = local.get(name, "")
        if isinstance(configured, str) and configured.strip():
            path = Path(configured).expanduser()
            path = path if path.is_absolute() else project_dir / path
            if path.is_dir():
                result[name] = path.resolve()
                continue
        result[name] = next((path for path in candidates if path.is_dir()), candidates[0])
    return result


def find_paper_reference_figures(project_dir: Path) -> Path | None:
    """Prefer the current paper assets, with the original folder as fallback."""
    workspace=Path(project_dir).resolve().parent
    for relative in ["paper_figures/reference", "figures"]:
        for paper in sorted(workspace.glob("Workmode*")):
            candidate=paper/relative
            if candidate.is_dir() and any(candidate.glob("*.png")):
                return candidate
    return None
