from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image

from .constants import VIEW_SPECS
from .datasets import scan_s3dis_annotations, select_s3dis

ProgressCallback = Callable[[int, int, str], None]


def read_s3dis_points(source_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(source_path)
    values = np.loadtxt(path, dtype=np.float32, ndmin=2)
    if values.shape[0] == 0 or values.shape[1] < 6:
        raise ValueError(f"Invalid S3DIS annotation (at least XYZRGB expected): {path}")
    points = values[:, :3]
    colors = np.clip(values[:, 3:6], 0, 255).astype(np.uint8)
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if len(points) == 0:
        raise ValueError(f"The S3DIS annotation contains no finite points: {path}")
    return points, colors


def aabb_features(points: np.ndarray) -> dict[str, float | int]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    extent = maximum - minimum
    width, depth, height = (float(value) for value in extent)
    volume = width * depth * height
    return {
        "width": width,
        "depth": depth,
        "height": height,
        "volume": volume,
        "xy_area": width * depth,
        "width_depth_ratio": width / depth if depth > 0 else 0.0,
        "height_width_ratio": height / width if width > 0 else 0.0,
        "height_depth_ratio": height / depth if depth > 0 else 0.0,
        "point_count": int(len(points)),
        "min_x": float(minimum[0]),
        "min_y": float(minimum[1]),
        "min_z": float(minimum[2]),
        "max_x": float(maximum[0]),
        "max_y": float(maximum[1]),
        "max_z": float(maximum[2]),
    }


def deterministic_subsample(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce render points uniformly by source order without randomness."""

    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    indices = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int64)
    return points[indices], colors[indices]


def _camera_basis(azimuth: float, elevation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)
    forward = np.asarray(
        [
            np.cos(azimuth_rad) * np.cos(elevation_rad),
            np.sin(azimuth_rad) * np.cos(elevation_rad),
            np.sin(elevation_rad),
        ],
        dtype=np.float32,
    )
    forward /= np.linalg.norm(forward)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-7:
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    screen_up = np.cross(right, forward)
    screen_up /= np.linalg.norm(screen_up)
    return right, screen_up, forward


def render_rgb_isometric(
    points: np.ndarray,
    colors: np.ndarray,
    azimuth: float,
    elevation: float,
    image_size: int = 512,
    point_radius: int = 1,
    margin_fraction: float = 0.07,
) -> Image.Image:
    """Render an orthographic RGB point-cloud view without OpenGL/Open3D."""

    if image_size < 64:
        raise ValueError('Image size must be at least 64 pixels.')
    centered = points - ((points.min(axis=0) + points.max(axis=0)) / 2.0)
    right, screen_up, forward = _camera_basis(azimuth, elevation)
    projected_x = centered @ right
    projected_y = centered @ screen_up
    depth = centered @ forward
    projected_x -= (projected_x.min() + projected_x.max()) / 2.0
    projected_y -= (projected_y.min() + projected_y.max()) / 2.0

    x_range = float(np.ptp(projected_x))
    y_range = float(np.ptp(projected_y))
    span = max(x_range, y_range, 1e-8)
    usable = image_size * (1.0 - 2.0 * margin_fraction)
    scale = usable / span
    pixel_x = np.rint(projected_x * scale + (image_size - 1) / 2.0).astype(np.int32)
    pixel_y = np.rint(-projected_y * scale + (image_size - 1) / 2.0).astype(np.int32)
    pixel_x = np.clip(pixel_x, 0, image_size - 1)
    pixel_y = np.clip(pixel_y, 0, image_size - 1)

    flat = pixel_y.astype(np.int64) * image_size + pixel_x.astype(np.int64)
    # Sort each pixel from far to near and retain the nearest color.
    order = np.lexsort((depth, flat))
    sorted_flat = flat[order]
    group_end = np.r_[sorted_flat[1:] != sorted_flat[:-1], True]
    winners = order[group_end]

    canvas = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    x_values = pixel_x[winners]
    y_values = pixel_y[winners]
    winner_colors = colors[winners]
    radius = max(0, int(point_radius))
    offsets = [
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= max(1, radius * radius)
    ]
    if radius == 0:
        offsets = [(0, 0)]
    for dx, dy in offsets:
        shifted_x = x_values + dx
        shifted_y = y_values + dy
        valid = (
            (shifted_x >= 0)
            & (shifted_x < image_size)
            & (shifted_y >= 0)
            & (shifted_y < image_size)
        )
        canvas[shifted_y[valid], shifted_x[valid]] = winner_colors[valid]
    return Image.fromarray(canvas, mode="RGB")


def _render_signature(image_size: int, max_points: int, point_radius: int) -> str:
    payload = {
        "renderer": "numpy-orthographic-v3-no-seed",
        "image_size": image_size,
        "max_points": max_points,
        "point_radius": point_radius,
        "views": VIEW_SPECS,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def prepare_s3dis_views(
    s3dis_root: str | Path,
    output_root: str | Path,
    area: str | None = None,
    image_size: int = 512,
    max_points: int = 30_000,
    point_radius: int = 1,
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Render true-RGB S3DIS instance views and persist geometry metadata."""

    destination = Path(output_root).expanduser().resolve()
    views_root = destination / "views"
    manifest_path = destination / "s3dis_manifest.csv"
    destination.mkdir(parents=True, exist_ok=True)
    views_root.mkdir(parents=True, exist_ok=True)

    discovered = scan_s3dis_annotations(s3dis_root)
    if area and str(area).strip().casefold() != "all":
        requested_area = str(area).strip()
        discovered = discovered[discovered["area"].astype(str).str.casefold().eq(requested_area.casefold())].copy()
        if discovered.empty:
            available = ", ".join(sorted(scan_s3dis_annotations(s3dis_root)["area"].astype(str).unique()))
            raise ValueError(f"S3DIS-Area {requested_area} not found. Available: {available or 'none'}")
    selected = select_s3dis(discovered)
    signature = _render_signature(image_size, max_points, point_radius)

    existing = pd.read_csv(manifest_path).fillna("") if manifest_path.exists() else pd.DataFrame()
    existing_by_id = {
        str(row["sample_id"]): row
        for _, row in existing.iterrows()
        if str(row.get("status", "")) == "ok"
    }

    new_rows: list[dict[str, object]] = []
    total = len(selected)
    for position, (_, source_row) in enumerate(selected.iterrows(), start=1):
        sample_id = str(source_row["sample_id"])
        cached = existing_by_id.get(sample_id)
        cached_paths = [Path(str(cached.get(f"view_{index}", ""))) for index in range(4)] if cached is not None else []
        if (
            not overwrite
            and cached is not None
            and str(cached.get("render_signature", "")) == signature
            and all(path.is_file() for path in cached_paths)
        ):
            new_rows.append(cached.to_dict())
            if progress:
                progress(position, total, f"Wiederverwendet: {sample_id}")
            continue

        row = source_row.to_dict()
        row.update(
            {
                "status": "error",
                "error": "",
                "render_signature": signature,
                "image_size": image_size,
                "max_render_points": max_points,
                "point_radius": point_radius,
                "view_specs_json": json.dumps(VIEW_SPECS, sort_keys=True),
                "color_source": "original_s3dis_rgb_columns",
                "aabb_source": "full_object_point_cloud_before_render_subsampling",
            }
        )
        try:
            points, colors = read_s3dis_points(str(source_row["source_path"]))
            row.update(aabb_features(points))
            render_points, render_colors = deterministic_subsample(
                points,
                colors,
                max_points=max_points,
            )
            sample_dir = views_root.joinpath(*sample_id.split("/"))
            sample_dir.mkdir(parents=True, exist_ok=True)
            for index, view in enumerate(VIEW_SPECS):
                image = render_rgb_isometric(
                    render_points,
                    render_colors,
                    azimuth=float(view["azimuth"]),
                    elevation=float(view["elevation"]),
                    image_size=image_size,
                    point_radius=point_radius,
                )
                output_path = sample_dir / f"iso_{index}_{view['name']}.png"
                image.save(output_path, format="PNG", optimize=True)
                row[f"view_{index}"] = str(output_path.resolve())
            row["rendered_point_count"] = len(render_points)
            row["status"] = "ok"
        except Exception as exc:  # One malformed object must not abort a long preparation run.
            row["error"] = f"{type(exc).__name__}: {exc}"
        new_rows.append(row)
        if progress:
            progress(position, total, f"Vorbereitet: {sample_id}")

    new_frame = pd.DataFrame(new_rows)
    if not existing.empty:
        untouched = existing[~existing["sample_id"].astype(str).isin(set(new_frame.get("sample_id", [])))]
        combined = pd.concat([untouched, new_frame], ignore_index=True, sort=False)
    else:
        combined = new_frame
    if not combined.empty:
        combined = combined.sort_values(["label", "sample_id"]).reset_index(drop=True)
    combined.to_csv(manifest_path, index=False, encoding="utf-8")
    return new_frame, manifest_path
