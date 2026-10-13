from __future__ import annotations

import csv
import hashlib
import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from .constants import PROMPT_VERSION
from .datasets import image_paths_from_row, manifest_fingerprint
from .evaluation import evaluate_results, load_results_csv
from .providers import BaseProvider, build_prompt, create_provider, prompt_fingerprint

RunProgress = Callable[[int, int, str, dict[str, object]], None]


@dataclass(frozen=True)
class ModelTarget:
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""
    max_concurrent_requests: int = 1

    def public(self) -> dict[str, object]:
        values: dict[str, object] = {
            "provider": self.provider,
            "model": self.model,
            "max_concurrent_requests": max(1, int(self.max_concurrent_requests)),
        }
        if self.base_url:
            values["base_url"] = self.base_url
        return values


@dataclass(frozen=True)
class ExperimentSpec:
    dataset: str
    labels: tuple[str, ...]
    area_scope: str = "all"
    view_counts: tuple[int, ...] = (1, 2, 3, 4)
    conditions: tuple[str, ...] = ("rgb", "rgb_aabb")
    repeat_count: int = 1
    max_retries: int = 2
    request_delay_s: float = 0.0
    run_name: str = "experiment"

    def public(self) -> dict[str, object]:
        return asdict(self)


RESULT_FIELDS = [
    "task_id",
    "timestamp",
    "dataset",
    "sample_id",
    "true_label",
    "provider",
    "model",
    "model_version",
    "condition",
    "view_count",
    "repeat",
    "status",
    "pred_label",
    "correct",
    "latency_s",
    "error",
    "usage_json",
    "geometry_json",
    "image_paths_json",
    "prompt_version",
    "prompt_hash",
    "prompt_text",
    "raw_response",
]


def _slug(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in cleaned.split("-") if part)[:48] or "experiment"


def create_run_directory(
    runs_root: str | Path,
    spec: ExperimentSpec,
    targets: Sequence[ModelTarget],
    records: pd.DataFrame,
) -> Path:
    destination = Path(runs_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fingerprint_columns = [column for column in ("sample_id", "label", "view_count") if column in records]
    fingerprint = manifest_fingerprint(records, fingerprint_columns)[:8]
    run_dir = destination / f"{timestamp}_{_slug(spec.run_name)}_{fingerprint}"
    suffix = 1
    while run_dir.exists():
        run_dir = destination / f"{timestamp}_{_slug(spec.run_name)}_{fingerprint}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "experiment": spec.public(),
        "targets": [target.public() for target in targets],
        "selected_rows": len(records),
        "selected_objects": int(records["sample_id"].nunique()),
        "manifest_fingerprint": fingerprint,
        "note": "API keys are intentionally never persisted.",
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    records.to_csv(run_dir / "samples.csv", index=False, encoding="utf-8")
    return run_dir


def _tasks(records: pd.DataFrame, spec: ExperimentSpec, targets: Sequence[ModelTarget]) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for target in targets:
        for repeat in range(1, spec.repeat_count + 1):
            if spec.dataset == "modelnet":
                subset = records[records["view_count"].isin(spec.view_counts)]
                for _, row in subset.iterrows():
                    view_count = int(row["view_count"])
                    tasks.append(
                        {
                            "target": target,
                            "row": row,
                            "repeat": repeat,
                            "view_count": view_count,
                            "condition": f"views_{view_count}",
                        }
                    )
            elif spec.dataset == "s3dis":
                for condition in spec.conditions:
                    if condition not in ("rgb", "rgb_aabb"):
                        raise ValueError(f"Unbekannte S3DIS-Bedingung: {condition}")
                    for _, row in records.iterrows():
                        tasks.append(
                            {
                                "target": target,
                                "row": row,
                                "repeat": repeat,
                                "view_count": 4,
                                "condition": condition,
                            }
                        )
            else:
                raise ValueError(f"Unbekannter Datensatz: {spec.dataset}")
    return tasks


def planned_request_count(records: pd.DataFrame, spec: ExperimentSpec, targets: Sequence[ModelTarget]) -> int:
    return len(_tasks(records, spec, targets))


def _task_id(task: dict[str, object], spec: ExperimentSpec) -> str:
    target: ModelTarget = task["target"]
    row: pd.Series = task["row"]
    payload = "|".join(
        [
            PROMPT_VERSION,
            spec.dataset,
            str(row["sample_id"]),
            target.provider,
            target.model,
            target.base_url,
            str(task["condition"]),
            str(task["view_count"]),
            str(task["repeat"]),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _execute_task(
    task: dict[str, object],
    spec: ExperimentSpec,
    provider: BaseProvider,
) -> dict[str, object]:
    identifier = _task_id(task, spec)
    target: ModelTarget = task["target"]
    row: pd.Series = task["row"]
    view_count = int(task["view_count"])
    condition = str(task["condition"])
    image_paths = image_paths_from_row(row, view_count)
    geometry = None
    if spec.dataset == "s3dis" and condition == "rgb_aabb":
        geometry = {name: float(row[name]) for name in ("width", "depth", "height")}
    prompt_text = build_prompt(spec.dataset, spec.labels, view_count, geometry=geometry)
    base_result: dict[str, object] = {
        "task_id": identifier,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": spec.dataset,
        "sample_id": row["sample_id"],
        "true_label": row["label"],
        "provider": target.provider,
        "model": target.model,
        "model_version": "",
        "condition": condition,
        "view_count": view_count,
        "repeat": task["repeat"],
        "status": "error",
        "pred_label": "",
        "correct": "",
        "latency_s": "",
        "error": "",
        "usage_json": "{}",
        "geometry_json": json.dumps(geometry or {}, sort_keys=True),
        "image_paths_json": json.dumps([str(path) for path in image_paths]),
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_fingerprint(prompt_text),
        "prompt_text": prompt_text,
        "raw_response": "",
    }
    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(spec.max_retries + 1):
        try:
            result = provider.classify(image_paths, prompt_text, spec.labels)
            base_result.update(
                {
                    "status": "ok",
                    "pred_label": result.label,
                    "correct": result.label == str(row["label"]),
                    "model_version": result.model_version,
                    "usage_json": json.dumps(result.usage, ensure_ascii=False, sort_keys=True),
                    "raw_response": result.raw_response,
                    "error": "",
                }
            )
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            if attempt < spec.max_retries:
                time.sleep(min(2**attempt, 30))
    base_result["latency_s"] = round(time.perf_counter() - started, 6)
    if last_error is not None:
        base_result["error"] = f"{type(last_error).__name__}: {last_error}"
    return base_result


def run_experiment(
    records: pd.DataFrame,
    spec: ExperimentSpec,
    targets: Sequence[ModelTarget],
    run_dir: str | Path,
    progress: RunProgress | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    destination = Path(run_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results_path = destination / "results.csv"
    existing = load_results_csv(results_path)
    completed = set(existing.loc[existing.get("status", "") == "ok", "task_id"].astype(str)) if not existing.empty else set()
    tasks = _tasks(records, spec, targets)
    providers = {
        (target.provider, target.model, target.base_url): create_provider(
            target.provider,
            target.model,
            target.api_key,
            target.base_url,
        )
        for target in targets
    }

    write_header = not results_path.exists() or results_path.stat().st_size == 0
    done = sum(1 for task in tasks if _task_id(task, spec) in completed)
    with results_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
            stream.flush()

        for target in targets:
            target_tasks = [
                task
                for task in tasks
                if task["target"] is target and _task_id(task, spec) not in completed
            ]
            if not target_tasks:
                continue
            provider = providers[(target.provider, target.model, target.base_url)]
            worker_count = max(1, int(target.max_concurrent_requests))
            task_iterator = iter(target_tasks)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                active: dict[Future[dict[str, object]], dict[str, object]] = {}

                def submit_next() -> bool:
                    try:
                        next_task = next(task_iterator)
                    except StopIteration:
                        return False
                    future = executor.submit(_execute_task, next_task, spec, provider)
                    active[future] = next_task
                    return True

                for worker_index in range(worker_count):
                    if worker_index > 0 and spec.request_delay_s > 0:
                        time.sleep(spec.request_delay_s)
                    if not submit_next():
                        break

                while active:
                    finished, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
                    for future in finished:
                        task = active.pop(future)
                        base_result = future.result()
                        writer.writerow({field: base_result.get(field, "") for field in RESULT_FIELDS})
                        stream.flush()
                        done += 1
                        if progress:
                            progress(
                                done,
                                len(tasks),
                                f"{target.provider}/{target.model}: "
                                f"{base_result['sample_id']} - {base_result['condition']}",
                                dict(base_result),
                            )
                        if spec.request_delay_s > 0:
                            time.sleep(spec.request_delay_s)
                        submit_next()

    results = load_results_csv(results_path)
    evaluated = evaluate_results(results, destination)
    return results, evaluated
