from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from mmfm_tool.rendering import aabb_features, prepare_s3dis_views, render_rgb_isometric


def test_rgb_renderer_keeps_content_inside_margin_and_aabb_is_exact() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
            [2.0, 3.0, 4.0],
        ],
        dtype=np.float32,
    )
    colors = np.asarray(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [30, 40, 50], [90, 100, 110]],
        dtype=np.uint8,
    )
    features = aabb_features(points)
    assert features["width"] == 2.0
    assert features["depth"] == 3.0
    assert features["height"] == 4.0
    assert features["volume"] == 24.0

    image = render_rgb_isometric(points, colors, 45.0, 35.264, image_size=128, point_radius=1)
    pixels = np.asarray(image)
    foreground = np.any(pixels != 255, axis=2)
    y_values, x_values = np.where(foreground)
    assert len(x_values) > 0
    assert x_values.min() > 0 and x_values.max() < 127
    assert y_values.min() > 0 and y_values.max() < 127
    assert len(np.unique(pixels[foreground], axis=0)) >= 3


def test_area5_preparation_persists_true_rgb_views_and_full_cloud_aabb(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "source" / "nested" / "Area_5" / "office_1" / "Annotations"
    annotation_dir.mkdir(parents=True)
    (annotation_dir / "chair_1.txt").write_text(
        "\n".join(
            [
                "0 0 0 255 0 0",
                "2 0 0 0 255 0",
                "0 3 0 0 0 255",
                "0 0 4 30 40 50",
                "2 3 4 90 100 110",
            ]
        ),
        encoding="utf-8",
    )

    prepared, manifest_path = prepare_s3dis_views(
        tmp_path / "source",
        tmp_path / "prepared",
        area="Area_5",
        image_size=128,
        max_points=3,
        point_radius=1,
    )

    assert manifest_path.is_file()
    assert len(prepared) == 1
    row = prepared.iloc[0]
    assert row["status"] == "ok"
    assert row["area"] == "Area_5"
    assert row["width"] == 2.0
    assert row["depth"] == 3.0
    assert row["height"] == 4.0
    assert row["point_count"] == 5
    assert row["rendered_point_count"] == 3
    assert row["color_source"] == "original_s3dis_rgb_columns"
    assert row["aabb_source"] == "full_object_point_cloud_before_render_subsampling"
    for index in range(4):
        image_path = Path(row[f"view_{index}"])
        assert image_path.is_file()
        pixels = np.asarray(Image.open(image_path).convert("RGB"))
        assert np.any(np.any(pixels != 255, axis=2))
