from __future__ import annotations

from pathlib import Path

from PIL import Image

import pandas as pd

from mmfm_tool.datasets import (
    find_s3dis_area_root,
    scan_modelnet_manifest,
    scan_s3dis_annotations,
    select_paired_modelnet,
    select_s3dis_area,
)


def _png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def test_modelnet_one_view_is_derived_from_the_same_test_object(tmp_path: Path) -> None:
    root = tmp_path / "Modelnet_TEST_1"
    # This mirrors the supplied layout: the one-view tree contains training only.
    _png(root / "Model_IMG_1_Isometrien" / "chair" / "train" / "chair_0001_iso_0.png", (0, 0, 0))
    for view_count in (2, 3, 4):
        for index in range(view_count):
            _png(
                root / f"Model_IMG_{view_count}_Isometrien" / "chair" / "test" / f"chair_0100_iso_{index}.png",
                (index * 20, 0, 0),
            )

    manifest = scan_modelnet_manifest(root)
    assert len(manifest) == 4
    assert manifest["sample_id"].nunique() == 1
    assert set(manifest["view_count"]) == {1, 2, 3, 4}
    one_view = manifest[manifest["view_count"] == 1].iloc[0]
    assert one_view["source_split"] == "test:derived_from_2_view"
    assert "Model_IMG_2_Isometrien" in one_view["view_0"]
    assert "chair_0100_iso_0.png" in one_view["view_0"]

    paired = select_paired_modelnet(manifest, [1, 2, 3, 4])
    assert len(paired) == 4
    assert paired["sample_id"].nunique() == 1


def test_s3dis_area_selection_keeps_every_object_in_the_selected_area() -> None:
    manifest = pd.DataFrame(
        [
            {"sample_id": "Area_1/office_1/chair_1", "label": "chair", "area": "Area_1"},
            {"sample_id": "Area_1/office_2/table_1", "label": "table", "area": "Area_1"},
            {"sample_id": "Area_2/office_1/chair_2", "label": "chair", "area": "Area_2"},
        ]
    )

    selected = select_s3dis_area(manifest, "Area_1")

    assert len(selected) == 2
    assert set(selected["area"]) == {"Area_1"}
    assert set(selected["sample_id"]) == {
        "Area_1/office_1/chair_1",
        "Area_1/office_2/table_1",
    }


def test_s3dis_layout_accepts_nested_area5_without_other_areas(tmp_path: Path) -> None:
    root = tmp_path / "S3DIS_TEST"
    area_root = root / "Stanford3dDataset_v1.2_Aligned_Version" / "nested"
    annotation_dir = area_root / "Area_5" / "office_1" / "Annotations"
    annotation_dir.mkdir(parents=True)
    (annotation_dir / "chair_1.txt").write_text("0 0 0 255 10 20\n1 2 3 30 200 40\n", encoding="utf-8")

    assert find_s3dis_area_root(root) == area_root.resolve()
    manifest = scan_s3dis_annotations(root)

    assert len(manifest) == 1
    assert manifest.iloc[0]["area"] == "Area_5"
    assert manifest.iloc[0]["sample_id"] == "Area_5/office_1/chair_1"
