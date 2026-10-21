from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import pandas as pd

from .providers import JevProvider, photo_check_scores, prompt_fingerprint


def run_classification_preview(
    image_paths: Sequence[Path],
    dataset: str,
    labels: Sequence[str],
    model: str,
    api_key: str,
    output_root: Path,
    *,
    geometry: dict[str, float] | None = None,
    sample_id: str = "",
    true_label: str = "",
) -> tuple[dict[str, object], Path]:
    """Run one small diagnostic request, saved outside the scientific run folders."""
    if len(image_paths) not in (1, 2):
        raise ValueError("Der Bildtest benötigt genau eine oder zwei Ansichten desselben Objekts.")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    provider = JevProvider(model, api_key)
    prompt = provider.build_prompt(dataset, labels, len(image_paths), geometry=geometry)
    result = provider.classify(image_paths, prompt, labels)
    record: dict[str, object] = {
        "preview_only": True, "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset, "sample_id": sample_id, "true_label": true_label,
        "provider": provider.provider_name, "model": model, "model_version": result.model_version,
        "status": "ok", "pred_label": result.label, "view_count": len(image_paths),
        "image_paths_json": json.dumps([str(path) for path in image_paths]),
        "geometry_json": json.dumps(geometry or {}), "prompt_text": prompt,
        "prompt_hash": prompt_fingerprint(prompt), "questions": provider.build_questions(prompt, labels),
        "usage_json": json.dumps(result.usage), "raw_response": result.raw_response,
    }
    filename = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{uuid4().hex[:8]}.json"
    destination = output_root / filename
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record, destination


# Compatibility for callers of the former image-test helper; new tests use Choice too.
run_photo_check_preview = run_classification_preview


def photo_check_table(results: pd.DataFrame) -> pd.DataFrame:
    """Expand valid photo checks into one auditable row per object and class."""
    metadata = ["task_id", "dataset", "sample_id", "model", "model_version", "condition",
                "view_count", "repeat", "true_label", "pred_label", "prompt_hash"]
    columns = [*metadata, "candidate_class", "yes_probability", "selected"]
    rows: list[dict[str, object]] = []
    if results.empty:
        return pd.DataFrame(columns=columns)
    subset = results[(results["provider"] == "jev_photo_check") & (results["status"] == "ok")]
    for _, result in subset.iterrows():
        scores = photo_check_scores(str(result["raw_response"]))
        for label, score in scores.items():
            rows.append({**{key: result.get(key, "") for key in metadata},
                         "candidate_class": label, "yes_probability": score,
                         "selected": label == result["pred_label"]})
    return pd.DataFrame(rows, columns=columns)
