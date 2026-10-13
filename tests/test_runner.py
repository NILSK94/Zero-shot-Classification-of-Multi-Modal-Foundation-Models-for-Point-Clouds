from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from mmfm_tool.providers import ProviderResult
from mmfm_tool.runner import ExperimentSpec, ModelTarget, create_run_directory, run_experiment


def test_model_target_public_persists_endpoint_but_never_api_key() -> None:
    target = ModelTarget(
        "litellm",
        "qwen3.6-35b-a3b-awq",
        api_key="must-not-be-written",
        base_url="https://proxy.example.test/v1",
    )

    assert target.public() == {
        "provider": "litellm",
        "model": "qwen3.6-35b-a3b-awq",
        "max_concurrent_requests": 1,
        "base_url": "https://proxy.example.test/v1",
    }
    assert "must-not-be-written" not in json.dumps(target.public())


def test_mock_s3dis_run_writes_raw_metrics_and_paired_comparison(tmp_path: Path) -> None:
    rows = []
    for sample_index, label in enumerate(("chair", "table")):
        row = {
            "dataset": "s3dis",
            "sample_id": f"Area_1/office_1/{label}_{sample_index + 1}",
            "label": label,
            "width": 0.5 + sample_index,
            "depth": 0.6 + sample_index,
            "height": 0.7 + sample_index,
        }
        for view_index in range(4):
            image_path = tmp_path / "images" / f"{sample_index}_{view_index}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (48, 48), (50 * sample_index, 30 * view_index, 120)).save(image_path)
            row[f"view_{view_index}"] = str(image_path)
        rows.append(row)
    records = pd.DataFrame(rows)
    spec = ExperimentSpec(
        dataset="s3dis",
        labels=("chair", "table"),
        view_counts=(4,),
        conditions=("rgb", "rgb_aabb"),
        repeat_count=1,
        run_name="qa",
    )
    targets = [ModelTarget("mock", "mock-v1")]
    run_dir = create_run_directory(tmp_path / "runs", spec, targets, records)
    progress_rows: list[dict[str, object]] = []
    results, evaluated = run_experiment(
        records,
        spec,
        targets,
        run_dir,
        progress=lambda done, total, message, result: progress_rows.append(result),
    )

    assert len(results) == 4
    assert set(results["status"]) == {"ok"}
    assert set(results["correct"].astype(str)) <= {"True", "False"}
    assert set(results["condition"]) == {"rgb", "rgb_aabb"}
    assert (run_dir / "config.json").exists()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert "seed" not in config["experiment"]
    assert (run_dir / "samples.csv").exists()
    assert (run_dir / "summary_aggregate.csv").exists()
    assert len(evaluated["aggregate"]) == 2
    assert len(evaluated["paired"]) == 1
    assert int(evaluated["paired"].iloc[0]["n_paired"]) == 2
    assert len(progress_rows) == 4
    assert {str(row["status"]) for row in progress_rows} == {"ok"}


def test_target_concurrency_setting_enforces_one_request_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    records_rows: list[dict[str, object]] = []
    for sample_index in range(4):
        image_path = tmp_path / "modelnet" / f"chair_{sample_index:04d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 24), (sample_index * 20, 40, 80)).save(image_path)
        records_rows.append(
            {
                "dataset": "modelnet",
                "sample_id": f"chair/chair_{sample_index:04d}",
                "label": "chair",
                "view_count": 1,
                "view_0": str(image_path),
            }
        )
    records = pd.DataFrame(records_rows)
    spec = ExperimentSpec(
        dataset="modelnet",
        labels=("chair", "table"),
        view_counts=(1,),
        conditions=(),
        max_retries=0,
        run_name="sequential-qwen",
    )

    class TrackingProvider:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def classify(self, image_paths: object, prompt: str, labels: object) -> ProviderResult:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.03)
            with self.lock:
                self.active -= 1
            return ProviderResult(
                label="chair",
                raw_response="chair",
                model_version="qwen-test",
                usage={},
            )

    provider = TrackingProvider()
    monkeypatch.setattr("mmfm_tool.runner.create_provider", lambda *args, **kwargs: provider)
    target = ModelTarget(
        "litellm",
        "qwen-test",
        api_key="test-only",
        base_url="http://proxy.invalid/v1",
        max_concurrent_requests=1,
    )
    run_dir = create_run_directory(tmp_path / "runs", spec, [target], records)
    run_experiment(records, spec, [target], run_dir)

    assert provider.max_active == 1

    parallel_provider = TrackingProvider()
    monkeypatch.setattr("mmfm_tool.runner.create_provider", lambda *args, **kwargs: parallel_provider)
    parallel_target = ModelTarget(
        "litellm",
        "qwen-test",
        api_key="test-only",
        base_url="http://proxy.invalid/v1",
        max_concurrent_requests=2,
    )
    parallel_run_dir = create_run_directory(tmp_path / "runs", spec, [parallel_target], records)
    run_experiment(records, spec, [parallel_target], parallel_run_dir)

    assert parallel_provider.max_active == 2
