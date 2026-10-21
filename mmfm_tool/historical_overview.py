"""A reproducible 3 x 4 historical overview, matching the original experiment layout."""
from __future__ import annotations

import json
import io
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .confusion_analysis import MatrixSettings, make_reference, matrix_values, prepare_confusion_rows


def overview_panels(
    archive: pd.DataFrame,
    settings: MatrixSettings,
    references: dict[str, tuple[MatrixSettings, dict]] | None = None,
    real_model: str = "gemini-1.5-pro",
) -> tuple[list[dict[str, Any]], dict[str, tuple[MatrixSettings, dict]]]:
    references = references or {}
    prepared: dict[str, pd.DataFrame] = {}
    resolved: dict[str, tuple[MatrixSettings, dict]] = {}
    for dataset in ("modelnet", "blk2go"):
        data = prepare_confusion_rows(archive, dataset, paired=False) if not archive.empty else pd.DataFrame()
        prepared[dataset] = data
        if data.empty:
            continue
        active, reference = references.get(dataset, (settings, None))
        active = replace(active, include_invalid=active.include_invalid and bool(data["prediction"].eq("[No valid response]").any()))
        if reference is None:
            # A manual selection or custom taxonomy for ModelNet cannot be applied to BLK2GO.
            active = replace(active, focus="all" if active.focus == "manual" else active.focus,
                             grouping="none" if active.grouping == "custom" else active.grouping)
            reference = make_reference(data, active)
        else:
            labels = set(data["true_label"]) | (set(data["prediction"]) - {"[No valid response]"})
            extra = sorted(labels - set(reference["order"]))
            reference = {**reference, "order": [*reference["order"], *extra],
                         "shown": [*reference["shown"], *extra],
                         "groups": {**reference["groups"], **{label: label for label in extra}}}
        resolved[dataset] = active, reference
    panels: list[dict[str, Any]] = []

    def add(row, col, dataset, model, scope, condition, title, placeholder="Not evaluated"):
        source = prepared[dataset]
        subset = (source[source["model"].eq(model) & source["protocol_scope"].eq(scope)
                         & source["condition"].eq(condition)] if not source.empty else pd.DataFrame())
        panel = {"row": row, "col": col, "dataset": dataset, "model": model, "scope": scope,
                 "condition": condition, "title": title, "data": subset, "placeholder": placeholder,
                 "counting": archive.attrs.get("counting", "object_decisions") if dataset == "modelnet" else "object_decisions"}
        if not subset.empty:
            active, reference = resolved[dataset]
            panel["matrix"] = matrix_values(subset, reference, active)
            panel["objects"] = int(subset["sample_id"].nunique())
            panel["observations"] = len(subset)
            panel["source_files"] = sorted(subset["source_run"].astype(str).unique().tolist()) if "source_run" in subset else []
        panels.append(panel)

    for row, model in [(1, "gemini-1.5-pro"), (2, "gpt-4o")]:
        for views in range(1, 5):
            add(row, views, "modelnet", model, "historical_modelnet_pilot", f"views_{views}",
                f"{views} view{'s' if views > 1 else ''}")
    add(3, 1, "modelnet", "stable-diffusion", "historical_modelnet_full", "views_1", "Stable Diffusion",
        "Not reconstructed")
    add(3, 2, "modelnet", "gemini-1.5-pro", "historical_modelnet_full", "views_1", "Gemini 1.5 Pro · full test")
    display_model = "Gemini 1.5 Pro" if real_model == "gemini-1.5-pro" else "GPT-4o"
    add(3, 3, "blk2go", real_model, "historical_blk2go_pilot", "rgb", f"{display_model} · BLK2GO RGB")
    add(3, 4, "blk2go", real_model, "historical_blk2go_pilot", "rgb_aabb", f"{display_model} · BLK2GO RGB + AABB")
    return panels, resolved


def overview_values(panels: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for panel in panels:
        if "matrix" not in panel:
            continue
        matrix = panel["matrix"]
        for truth in matrix["counts"].index:
            for prediction in matrix["counts"].columns:
                count = int(matrix["counts"].loc[truth, prediction])
                correct = int(matrix["correct"].loc[truth, prediction])
                records.append({"row": panel["row"], "column": panel["col"], "dataset": panel["dataset"],
                                "model": panel["model"], "protocol_scope": panel["scope"], "condition": panel["condition"],
                                "objects": panel["objects"], "observations": panel["observations"], "counting": panel["counting"],
                                "true_group": truth, "predicted_group": prediction,
                                "count": count, "original_correct": correct, "original_errors": count-correct,
                                "row_support": int(matrix["support"].loc[truth]),
                                "display_value": matrix["values"].loc[truth, prediction]})
    return pd.DataFrame(records)


def _cell_text(value: float, normalized: bool) -> str:
    return f"{value * 100:.0f}" if normalized else f"{value:.0f}"


def historical_overview_figure(panels: list[dict[str, Any]], references: dict[str, tuple[MatrixSettings, dict]],
                               *, show_values: bool = True, show_zeros: bool = False) -> go.Figure:
    titles = [panel["title"] for panel in panels]
    figure = make_subplots(rows=3, cols=4, subplot_titles=titles, horizontal_spacing=.065, vertical_spacing=.12)
    for panel in panels:
        row, col = panel["row"], panel["col"]
        axis_number = (row - 1) * 4 + col
        suffix = "" if axis_number == 1 else str(axis_number)
        if "matrix" not in panel:
            figure.add_annotation(text=panel["placeholder"], x=.5, y=.5,
                                  xref=f"x{suffix} domain", yref=f"y{suffix} domain", showarrow=False,
                                  font={"size": 15, "color": "#64748b"})
            figure.update_xaxes(showticklabels=False, showgrid=False, range=[0, 1], row=row, col=col)
            figure.update_yaxes(showticklabels=False, showgrid=False, range=[0, 1], row=row, col=col)
            continue
        matrix = panel["matrix"]
        active, reference = references[panel["dataset"]]
        x = [matrix["names"].get(name, name) for name in matrix["counts"].columns]
        y = [matrix["names"].get(name, name) for name in matrix["counts"].index]
        counts = matrix["counts"].to_numpy()
        correct = matrix["correct"].to_numpy()
        denominator = np.repeat(matrix["support"].to_numpy()[:, None], len(x), axis=1)
        coloraxis = "coloraxis" if panel["dataset"] == "modelnet" else "coloraxis2"
        figure.add_trace(go.Heatmap(x=x, y=y, z=matrix["displayed"].to_numpy(), coloraxis=coloraxis,
                                   customdata=np.stack([counts, correct, counts-correct, denominator], axis=-1),
                                   hoverongaps=False,
                                   hovertemplate="True: %{y}<br>Predicted: %{x}<br>Value: %{z:.3f}<br>Count: %{customdata[0]}<br>Originally correct: %{customdata[1]}<br>Originally wrong: %{customdata[2]}<br>Denominator: %{customdata[3]}<extra></extra>"), row=row, col=col)
        if show_values:
            tx, ty, texts, colors = [], [], [], []
            maximum = 1. if active.normalize else max(1, reference.get("count_max", 1))
            for yi, row_name in enumerate(y):
                for xi, column_name in enumerate(x):
                    value = matrix["displayed"].iloc[yi, xi]
                    if not np.isfinite(value) or (not show_zeros and value == 0):
                        continue
                    tx.append(column_name); ty.append(row_name)
                    texts.append(_cell_text(value, active.normalize))
                    colors.append("white" if value / maximum > .55 else "#1e293b")
            figure.add_trace(go.Scatter(x=tx, y=ty, mode="text", text=texts,
                                       textfont={"size": 8, "color": colors}, hoverinfo="skip", showlegend=False), row=row, col=col)
        figure.update_xaxes(categoryorder="array", categoryarray=x, title_text="Predicted" if row == 3 else "",
                            tickangle=90, tickfont={"size": 9}, showgrid=False, row=row, col=col)
        figure.update_yaxes(categoryorder="array", categoryarray=y, autorange="reversed",
                            showticklabels=col == 1 or row == 3,
                            title_text="True" if col == 1 else "", tickfont={"size": 9}, showgrid=False, row=row, col=col)
    for row, text in [(1, "Gemini 1.5 Pro"), (2, "GPT-4o")]:
        suffix = "" if row == 1 else "5"
        figure.add_annotation(text=f"<b>{text}</b>", x=-.075, y=.5, xref="paper", yref=f"y{suffix} domain",
                              textangle=-90, showarrow=False, font={"size": 20})
    for dataset, axis_name, y_position, length in [("modelnet", "coloraxis", .66, .62), ("blk2go", "coloraxis2", .14, .23)]:
        if dataset not in references:
            continue
        active, ref = references[dataset]
        figure.update_layout(**{axis_name: {"colorscale": "Blues", "cmin": 0,
            "cmax": 1. if active.normalize else max(1, ref.get("count_max", 1)),
            "colorbar": {"title": f"{dataset.upper()}<br>{'Row share' if active.normalize else 'Count'}",
                         "tickformat": ".0%" if active.normalize else "d", "y": y_position, "len": length, "thickness": 14}}})
    divider = figure.layout.yaxis9.domain[1] + .045
    figure.add_shape(type="line", x0=0, x1=1, y0=divider, y1=divider, xref="paper", yref="paper", line={"color": "#475569", "width": 2})
    note = "Counts / row %. Grey: no data."
    figure.add_annotation(text=note, x=0, y=0, xref="paper", yref="paper", yshift=-155,
                          xanchor="left", yanchor="top", align="left", showarrow=False, font={"size": 11})
    figure.update_layout(title={"text": "", "x": .02}, template="simple_white",
                         width=2150, height=2050, margin={"l": 160, "r": 140, "t": 100, "b": 235},
                         font={"family": "Arial", "size": 12}, paper_bgcolor="white", plot_bgcolor="#eeeeee")
    return figure


def save_historical_overview(output_root: Path, panels: list[dict[str, Any]],
                             references: dict[str, tuple[MatrixSettings, dict]],
                             figure: go.Figure, *, show_values: bool = True, show_zeros: bool = False) -> dict[str, Path]:
    """Save the actual matrix values and publication-sized SVG/PNG, not a screenshot."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import Normalize

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {kind: output_root / f"historical_overview.{kind}" for kind in ("html", "csv", "json", "svg", "png")}
    figure.write_html(paths["html"], include_plotlyjs=True)
    overview_values(panels).to_csv(paths["csv"], index=False)
    manifest = {"schema_version": 1, "references": {dataset: {**ref, "settings": settings.public()} for dataset, (settings, ref) in references.items()},
                "panels": [{key: panel.get(key) for key in ["row", "col", "dataset", "model", "scope", "condition", "objects", "observations", "counting", "source_files", "placeholder"]} for panel in panels],
                "show_values": show_values, "show_zeros": show_zeros}
    paths["json"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(3, 4, figsize=(25, 20))
    fig.subplots_adjust(left=.07, right=.925, top=.94, bottom=.105, wspace=.36, hspace=.7)
    maps = {}
    for panel in panels:
        row, col = panel["row"]-1, panel["col"]-1
        ax = axes[row, col]
        ax.set_title(panel["title"], fontsize=12, pad=12)
        if "matrix" not in panel:
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("#f1f5f9")
            ax.text(.5, .5, panel["placeholder"].replace("<br>", "\n"), ha="center", va="center", transform=ax.transAxes, fontsize=12, color="#64748b")
            continue
        matrix = panel["matrix"]
        active, ref = references[panel["dataset"]]
        maximum = 1. if active.normalize else max(1, ref.get("count_max", 1))
        values = matrix["displayed"].to_numpy()
        cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#eeeeee")
        maps[panel["dataset"]] = ax.imshow(values, cmap=cmap, norm=Normalize(0, maximum), aspect="equal")
        labels_x = [matrix["names"].get(name, name) for name in matrix["counts"].columns]
        labels_y = [matrix["names"].get(name, name) for name in matrix["counts"].index]
        font = 6.5 if len(labels_y) > 20 else 8
        ax.set_xticks(range(len(labels_x)), labels_x, rotation=90, fontsize=font)
        ax.set_yticks(range(len(labels_y)), labels_y if col == 0 or row == 2 else [""]*len(labels_y), fontsize=font)
        if row == 2:
            ax.set_xlabel("Predicted", fontsize=9)
        if col == 0:
            ax.set_ylabel("True", fontsize=9)
        if show_values:
            for yi in range(values.shape[0]):
                for xi in range(values.shape[1]):
                    value = values[yi, xi]
                    if np.isfinite(value) and (show_zeros or value != 0):
                        ax.text(xi, yi, _cell_text(value, active.normalize), ha="center", va="center",
                                fontsize=4.5 if len(labels_y) > 20 else 7,
                                color="white" if value / maximum > .55 else "#1e293b")
    fig.text(.015, .81, "Gemini 1.5 Pro", rotation=90, ha="center", va="center", fontsize=20)
    fig.text(.015, .49, "GPT-4o", rotation=90, ha="center", va="center", fontsize=20)
    from matplotlib.lines import Line2D
    divider = (axes[1, 0].get_position().y0 + axes[2, 0].get_position().y1) / 2. - .02
    fig.add_artist(Line2D([.05, .925], [divider, divider], transform=fig.transFigure, color="#475569", linewidth=1.5))
    for dataset, bottom, height in [("modelnet", .45, .44), ("blk2go", .14, .19)]:
        if dataset in maps:
            color_axis = fig.add_axes([.95, bottom, .008, height])
            colorbar = fig.colorbar(maps[dataset], cax=color_axis)
            active, _ = references[dataset]
            colorbar.set_label(f"{dataset.upper()}: {'row share' if active.normalize else 'count'}", fontsize=10)
    fig.text(.07, .025, "Counts / row %. Grey: no data.", fontsize=11)
    try:
        png = io.BytesIO()
        fig.savefig(png, format="png", dpi=200, facecolor="white")
        paths["png"].write_bytes(png.getvalue())
        fig.savefig(paths["svg"], facecolor="white")
    finally:
        plt.close(fig)
    return paths
