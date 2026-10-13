from __future__ import annotations

from pathlib import Path

import pandas as pd

from mmfm_tool.legacy_results import import_legacy_results
from mmfm_tool.paper_export import build_paper_export


def test_legacy_results_are_aligned_to_exact_current_ids_and_missing_rows_are_errors(tmp_path: Path) -> None:
    old_root = tmp_path / "Old_results"
    old_root.mkdir()
    legacy = pd.DataFrame(
        [
            {
                "sample_id": "chair_0001",
                "ground_truth": "chair",
                "predicted_label": "chair",
                "correct": 1,
                "confidence": "",
                "model_provider": "ollama",
                "model_name": "vision-test",
                "prompt_name": "vision_concise_v1",
                "num_views": 1,
                "feature_mode": "vision_only",
                "render_type": "isometric",
                "noise_level": 0.0,
                "run_index": 0,
                "input_tokens": 100,
                "output_tokens": 1,
                "estimated_cost_usd": 0.0,
                "runtime_seconds": 0.5,
                "raw_response": "chair",
            },
            {
                "sample_id": "bathtub_0001",
                "ground_truth": "bathtub",
                "predicted_label": "bathtub",
                "correct": 1,
                "confidence": "",
                "model_provider": "ollama",
                "model_name": "vision-test",
                "prompt_name": "vision_concise_v1",
                "num_views": 1,
                "feature_mode": "vision_only",
                "render_type": "isometric",
                "noise_level": 0.0,
                "run_index": 0,
                "input_tokens": 100,
                "output_tokens": 1,
                "estimated_cost_usd": 0.0,
                "runtime_seconds": 0.5,
                "raw_response": "bathtub",
            },
        ]
    )
    legacy.to_csv(old_root / "ollama__vision-test__vision_concise_v1__v1.csv", index=False)
    current_manifest = pd.DataFrame(
        [
            {"sample_id": "chair/chair_0001", "label": "chair", "view_count": 1},
            {"sample_id": "table/table_0001", "label": "table", "view_count": 1},
        ]
    )

    imported = import_legacy_results(old_root, current_manifest, tmp_path / "runs")
    comparable = pd.read_csv(imported["comparable33"] / "results.csv").fillna("")
    summary = pd.read_csv(imported["comparable33"] / "summary_by_repeat.csv")

    assert len(comparable) == 2
    assert set(comparable["sample_id"]) == {"chair/chair_0001", "table/table_0001"}
    assert set(comparable["true_label"]) == {"chair", "table"}
    assert comparable.loc[comparable["sample_id"] == "table/table_0001", "status"].iloc[0] == "error"
    assert int(summary["n_planned"].iloc[0]) == 2
    assert float(summary["coverage"].iloc[0]) == 0.5
    assert float(summary["end_to_end_accuracy"].iloc[0]) == 0.5

    paper_export = build_paper_export(tmp_path / "runs", tmp_path / "paper")
    paper_summary = pd.read_csv(paper_export.files["paper_model_view_summary"])

    assert paper_export.zip_path.exists()
    assert len(paper_summary) == 1
    assert paper_summary["paper_status"].iloc[0] == "Teillauf – nicht direkt vergleichbar"
