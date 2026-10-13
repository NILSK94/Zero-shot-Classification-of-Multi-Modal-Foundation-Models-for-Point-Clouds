from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def model_only_legend(figure: go.Figure) -> go.Figure:
    """Collapse Plotly Express compound legend names to the model only."""

    seen: set[str] = set()
    for trace in figure.data:
        model = str(trace.name).split(",", maxsplit=1)[0].strip()
        trace.name = model
        trace.legendgroup = model
        trace.showlegend = model not in seen
        seen.add(model)
    figure.update_layout(legend_title_text="Model")
    return figure


def confusion_matrix_grid(
    results: pd.DataFrame,
    *,
    model: str,
    normalize: bool = True,
) -> go.Figure:
    """Return one exportable figure containing ModelNet views 1–4 side by side."""

    planned_data = results.copy()
    planned_data = planned_data[
        planned_data["dataset"].astype(str).str.casefold().eq("modelnet")
        & planned_data["model"].astype(str).eq(str(model))
    ].copy()
    planned_data["view_count"] = pd.to_numeric(planned_data["view_count"], errors="coerce")
    planned_views = set(int(value) for value in planned_data["view_count"].dropna().unique())
    data = planned_data[planned_data["status"].astype(str).eq("ok")].copy()
    data["view_count"] = pd.to_numeric(data["view_count"], errors="coerce")
    labels = sorted(
        set(data["true_label"].dropna().astype(str))
        | set(data["pred_label"].dropna().astype(str))
    )
    available_views = sorted(int(value) for value in data["view_count"].dropna().unique() if 1 <= int(value) <= 4)
    titles = ["1 view", "2 views", "3 views", "4 views"]
    figure = make_subplots(rows=1, cols=4, subplot_titles=titles, shared_yaxes=True, horizontal_spacing=0.025)

    for view_count in range(1, 5):
        panel = data[data["view_count"].eq(view_count)]
        if panel.empty or not labels:
            axis_suffix = "" if view_count == 1 else str(view_count)
            figure.add_annotation(
                text="No valid responses" if view_count in planned_views else "Not evaluated",
                x=0.5,
                y=0.5,
                xref=f"x{axis_suffix} domain",
                yref=f"y{axis_suffix} domain",
                showarrow=False,
            )
            continue
        matrix = pd.crosstab(panel["true_label"], panel["pred_label"]).reindex(
            index=labels,
            columns=labels,
            fill_value=0,
        )
        values = matrix.astype(float)
        if normalize:
            row_totals = values.sum(axis=1).replace(0, pd.NA)
            values = values.div(row_totals, axis=0).fillna(0.0)
        figure.add_trace(
            go.Heatmap(
                z=values.to_numpy(),
                x=labels,
                y=labels,
                zmin=0,
                zmax=1 if normalize else None,
                colorscale="Blues",
                colorbar={"title": "Rate" if normalize else "Count", "tickformat": ".0%" if normalize else "d"},
                showscale=bool(available_views) and view_count == available_views[-1],
                hovertemplate=(
                    "True: %{y}<br>Predicted: %{x}<br>Rate: %{z:.1%}<extra></extra>"
                    if normalize
                    else "True: %{y}<br>Predicted: %{x}<br>Count: %{z:.0f}<extra></extra>"
                ),
            ),
            row=1,
            col=view_count,
        )

    figure.update_layout(
        title=f"{model} — confusion matrices by number of views (valid responses)",
        template="simple_white",
        height=760,
        margin={"l": 90, "r": 90, "t": 90, "b": 150},
    )
    for index in range(1, 5):
        x_name = "xaxis" if index == 1 else f"xaxis{index}"
        y_name = "yaxis" if index == 1 else f"yaxis{index}"
        figure.layout[x_name].update(title="Predicted class", tickangle=60, tickfont={"size": 8})
        figure.layout[y_name].update(
            title="True class" if index == 1 else "",
            tickfont={"size": 8},
            showticklabels=index == 1,
            autorange="reversed",
        )
    return figure
