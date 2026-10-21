from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


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
    settings=None,
    reference=None,
) -> go.Figure:
    """Use the same matrix renderer for current, legacy and single-run figures."""
    from dataclasses import replace
    from .confusion_analysis import MatrixSettings, confusion_figure, make_reference, prepare_confusion_rows

    selected = results[results["model"].astype(str).eq(str(model))]
    data = prepare_confusion_rows(selected, "modelnet", paired=False)
    active = replace(settings or MatrixSettings(), normalize=normalize)
    if reference is None:
        reference = make_reference(data, active)
    else:
        # A single audit run may contain extra classes not loaded into the explorer.
        # Keep existing positions and append every new label rather than drop observations.
        universe = set(data["true_label"]) | (set(data["prediction"]) - {"[No valid response]"})
        extra = sorted(universe - set(reference["order"]))
        reference = {**reference, "order": [*reference["order"], *extra],
                     "shown": [*reference["shown"], *extra],
                     "groups": {**reference["groups"], **{label: label for label in extra}}}
    return confusion_figure(data, reference, active,
                            title=f"{model} - confusion matrices by number of views",
                            expected_conditions=[f"views_{index}" for index in range(1, 5)])
