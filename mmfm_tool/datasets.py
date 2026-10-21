from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import pandas as pd

from .constants import S3DIS_CLASSES

ISO_PATTERN = re.compile(r"^(?P<object>.+)_iso_(?P<index>\d+)$", re.IGNORECASE)
INSTANCE_PATTERN = re.compile(r"^(?P<label>.+)_\d+$")
VIEW_COLUMNS = [f"view_{index}" for index in range(4)]


class DatasetLayoutError(ValueError):
    """Raised when a dataset does not match the expected local structure."""


def _view_root(modelnet_root: Path, view_count: int) -> Path:
    return modelnet_root / f"Model_IMG_{view_count}_Isometrien"


def _scan_modelnet_view(view_root: Path) -> dict[tuple[str, str], dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    if not view_root.exists():
        return grouped

    for class_dir in sorted(path for path in view_root.iterdir() if path.is_dir()):
        for split_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            for image_path in sorted(split_dir.glob("*.png")):
                match = ISO_PATTERN.match(image_path.stem)
                if not match:
                    continue
                key = (class_dir.name.lower(), match.group("object"))
                entry = grouped.setdefault(
                    key,
                    {"label": class_dir.name.lower(), "object_id": match.group("object"), "split": split_dir.name.lower(), "images": {}},
                )
                entry["images"][int(match.group("index"))] = image_path.resolve()
    return grouped


def scan_modelnet_manifest(
    modelnet_root: str | Path,
    view_counts: Sequence[int] = (1, 2, 3, 4),
) -> pd.DataFrame:
    """Create an object-level, paired manifest for the local ModelNet subset.

    The one-view directory in the supplied dataset stores all files below a
    ``train`` folder. Test membership is therefore inferred from the explicit
    test split in the two-view directory (or the first available multi-view
    directory). This prevents training objects from leaking into evaluation.
    """

    root = Path(modelnet_root).expanduser().resolve()
    if not root.exists():
        raise DatasetLayoutError(f"ModelNet folder not found: {root}")

    scans = {count: _scan_modelnet_view(_view_root(root, count)) for count in view_counts}
    reference_count = next(
        (
            count
            for count in sorted(scans)
            if any(str(entry["split"]) == "test" for entry in scans[count].values())
        ),
        None,
    )
    if reference_count is None:
        raise DatasetLayoutError('No explicit ModelNet test split found in the view folders.')

    test_keys = {
        key
        for key, entry in scans[reference_count].items()
        if str(entry["split"]) == "test"
    }
    rows: list[dict[str, object]] = []
    for view_count in sorted(set(int(value) for value in view_counts)):
        if view_count not in (1, 2, 3, 4):
            raise ValueError('This tool supports exactly 1 to 4 ModelNet views.')
        for key in sorted(test_keys):
            entry = scans[view_count].get(key)
            derived_one_view = False
            # The supplied one-view tree contains training files only. The paper's
            # one-view test used iso_0 from the multi-view test tree, which is also
            # the scientifically correct way to keep identical test object IDs.
            if view_count == 1 and entry is None:
                entry = scans[reference_count].get(key)
                derived_one_view = entry is not None
            if not entry:
                continue
            images = entry["images"]
            expected_indices = set(range(view_count))
            if not expected_indices.issubset(images):
                continue
            row: dict[str, object] = {
                "dataset": "modelnet",
                "sample_id": f"{key[0]}/{key[1]}",
                "label": key[0],
                "object_id": key[1],
                "view_count": view_count,
                "source_split": f"test:derived_from_{reference_count}_view" if derived_one_view else entry["split"],
            }
            for index, column in enumerate(VIEW_COLUMNS):
                row[column] = str(images[index]) if index < view_count else ""
            rows.append(row)

    columns = [
        "dataset",
        "sample_id",
        "label",
        "object_id",
        "view_count",
        "source_split",
        *VIEW_COLUMNS,
    ]
    return pd.DataFrame(rows, columns=columns)


def select_paired_modelnet(
    manifest: pd.DataFrame,
    view_counts: Sequence[int],
) -> pd.DataFrame:
    """Select every common test object for every requested view condition."""

    requested = sorted(set(int(value) for value in view_counts))
    subset = manifest[manifest["view_count"].isin(requested)].copy()
    if subset.empty:
        return subset

    presence = subset.groupby("sample_id")["view_count"].nunique()
    common_ids = set(presence[presence == len(requested)].index)
    objects = (
        subset[subset["sample_id"].isin(common_ids)][["sample_id", "label"]]
        .drop_duplicates()
        .sort_values(["label", "sample_id"])
    )

    selected_ids = set(objects["sample_id"])
    return (
        subset[subset["sample_id"].isin(selected_ids)]
        .sort_values(["view_count", "label", "sample_id"])
        .reset_index(drop=True)
    )


def find_s3dis_area_root(s3dis_root: str | Path) -> Path:
    root = Path(s3dis_root).expanduser().resolve()
    if not root.exists():
        raise DatasetLayoutError(f"S3DIS folder not found: {root}")

    valid_area_names = {f"Area_{index}" for index in range(1, 7)}
    candidates = {root}
    candidates.update(
        path.parent
        for path in root.rglob("Area_*")
        if path.is_dir() and path.name in valid_area_names
    )
    ranked: list[tuple[int, int, str, Path]] = []
    for candidate in candidates:
        area_count = sum((candidate / area).is_dir() for area in valid_area_names)
        if area_count:
            ranked.append((-area_count, len(candidate.parts), str(candidate), candidate))
    if ranked:
        return sorted(ranked)[0][-1]
    raise DatasetLayoutError(
        f"No Area_1 to Area_6 folder found under {root} found. "
        "A single available area directory, such as Area_5, is sufficient."
    )


def s3dis_label_from_path(path: Path) -> str:
    match = INSTANCE_PATTERN.match(path.stem)
    return (match.group("label") if match else path.stem).lower()


def scan_s3dis_annotations(s3dis_root: str | Path) -> pd.DataFrame:
    area_root = find_s3dis_area_root(s3dis_root)
    rows: list[dict[str, object]] = []
    for area_dir in sorted(area_root.glob("Area_*")):
        if not area_dir.is_dir():
            continue
        for room_dir in sorted(path for path in area_dir.iterdir() if path.is_dir()):
            annotation_dir = room_dir / "Annotations"
            if not annotation_dir.is_dir():
                continue
            for source_path in sorted(annotation_dir.glob("*.txt")):
                if source_path.stat().st_size == 0:
                    continue
                label = s3dis_label_from_path(source_path)
                rows.append(
                    {
                        "dataset": "s3dis",
                        "sample_id": f"{area_dir.name}/{room_dir.name}/{source_path.stem}",
                        "label": label,
                        "area": area_dir.name,
                        "room": room_dir.name,
                        "object_id": source_path.stem,
                        "source_path": str(source_path.resolve()),
                        "source_bytes": source_path.stat().st_size,
                        "known_class": label in S3DIS_CLASSES,
                    }
                )
    return pd.DataFrame(rows)


def select_s3dis(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return every discovered S3DIS instance in stable filesystem order."""

    return manifest.sort_values(["label", "area", "room", "sample_id"]).reset_index(drop=True)


def select_s3dis_area(manifest: pd.DataFrame, area: str = "all") -> pd.DataFrame:
    """Return the complete prepared cohort for one Area or for all Areas."""

    selected = str(area).strip()
    if not selected or selected.casefold() in {"all", "alle", "alle areas"}:
        return manifest.sort_values(["label", "area", "sample_id"]).reset_index(drop=True)
    if "area" not in manifest:
        raise DatasetLayoutError('The S3DIS manifest has no area column.')
    filtered = manifest[manifest["area"].astype(str).str.casefold() == selected.casefold()].copy()
    if filtered.empty:
        raise DatasetLayoutError(f"No prepared S3DIS instances found for {selected} found.")
    return filtered.sort_values(["label", "sample_id"]).reset_index(drop=True)


def load_prepared_s3dis_manifest(manifest_path: str | Path) -> pd.DataFrame:
    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path).fillna("")
    for column in ("width", "depth", "height", "volume", "point_count"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def image_paths_from_row(row: pd.Series | dict[str, object], view_count: int) -> list[Path]:
    paths: list[Path] = []
    for index in range(view_count):
        value = row.get(f"view_{index}", "")
        if not value:
            raise DatasetLayoutError(f"View {index} is missing for sample {row.get('sample_id', '?')}.")
        path = Path(str(value))
        if not path.exists():
            raise DatasetLayoutError(f"Image file is missing: {path}")
        paths.append(path)
    return paths


def manifest_fingerprint(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    import hashlib

    records = frame[list(columns)].sort_values(list(columns)).to_dict(orient="records")
    payload = json.dumps(records, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
