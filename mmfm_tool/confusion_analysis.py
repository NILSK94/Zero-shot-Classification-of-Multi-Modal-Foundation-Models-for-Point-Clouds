"""Shared, reproducible confusion analysis for current and imported result rows."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INVALID = "[No valid response]"
OTHER = "[Other classes]"
SERIES = ["protocol_scope", "provider", "model"]


@dataclass(frozen=True)
class MatrixSettings:
    focus: str = "all"
    top_k: int = 12
    min_support: int = 5
    stable_fraction: float = .95
    recall_tolerance: float = .05
    good_recall: float = .8
    bad_recall: float = .2
    order: str = "alphabetical"
    grouping: str = "none"
    clusters: int = 6
    normalize: bool = True
    errors_only: bool = False
    mask_below: float = 0.
    include_invalid: bool = True

    def public(self) -> dict[str, Any]:
        return asdict(self)


def prepare_confusion_rows(results: pd.DataFrame, dataset: str, paired: bool = True) -> pd.DataFrame:
    fields = ["dataset", "true_label", "pred_label", "status", "view_count", "condition", "repeat",
              *SERIES, "sample_id", "task_id", "source_run"]
    data = results.loc[results["dataset"].astype(str).str.casefold().eq(dataset.casefold()),
                       [field for field in fields if field in results]].copy()
    if data.empty:
        return data
    for column, default in [("protocol_scope", "current_paper_protocol"), ("provider", "unknown"), ("model", "unknown")]:
        if column not in data:
            data[column] = default
        data[column] = data[column].fillna(default).astype(str)
    if "task_id" in data:
        # Input merges already resolve success/newest precedence; do not count resumed rows twice.
        has_id = data["task_id"].fillna("").astype(str).ne("")
        data = pd.concat([data.loc[has_id].drop_duplicates(["protocol_scope", "provider", "model", "task_id"], keep="last"), data.loc[~has_id]])
    data["true_label"] = data["true_label"].fillna("").astype(str).str.casefold()
    data = data[data["true_label"].ne("")].copy()
    data["pred_label"] = data["pred_label"].fillna("").astype(str).str.casefold()
    valid = data["status"].astype(str).eq("ok") & data["pred_label"].ne("")
    data["prediction"] = data["pred_label"].where(valid, INVALID)
    data["is_correct"] = valid & data["true_label"].eq(data["pred_label"])
    data["view_count"] = pd.to_numeric(data["view_count"], errors="coerce")
    if "repeat" not in data:
        data["repeat"] = 1
    data["repeat"] = pd.to_numeric(data["repeat"], errors="coerce").fillna(1).astype(int)
    if "condition" not in data:
        data["condition"] = data["view_count"].map(lambda value: f"views_{int(value)}")
    data["condition"] = data["condition"].fillna("").astype(str)
    data["series_id"] = data[SERIES].astype(str).agg(" | ".join, axis=1)
    data["panel_id"] = data["series_id"] + " | " + data["condition"] + " | " + data["view_count"].astype(str)
    if paired:
        presence = data.groupby("sample_id")["panel_id"].nunique()
        common = presence[presence.eq(data["panel_id"].nunique())].index
        data = data[data["sample_id"].isin(common)].copy()
    return data.reset_index(drop=True)


def class_profiles(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["label", "min_support", "observations", "min_recall", "max_recall", "recall_range",
               "mean_recall", "unchanged_fraction", "paired_objects", "panels"]
    if data.empty:
        return pd.DataFrame(columns=columns)
    labels = sorted(set(data["true_label"]) | (set(data["prediction"]) - {INVALID}))
    panels = sorted(data["panel_id"].unique())
    totals = data.groupby(["panel_id", "true_label"]).size().unstack(fill_value=0).reindex(index=panels, columns=labels, fill_value=0)
    object_support = data.groupby(["panel_id", "true_label"])["sample_id"].nunique().unstack(fill_value=0).reindex(index=panels, columns=labels, fill_value=0)
    correct = data.groupby(["panel_id", "true_label"])["is_correct"].sum().unstack(fill_value=0).reindex(index=panels, columns=labels, fill_value=0)
    recall = correct.div(totals.replace(0, np.nan))
    # Identical aggregate recalls do not establish stability: compare object predictions too.
    objects = data.groupby(["sample_id", "true_label"], sort=True).agg(
        panels=("panel_id", "nunique"), predictions=("prediction", "nunique"),
        valid=("prediction", lambda values: bool(values.ne(INVALID).all())),
    ).reset_index()
    objects["paired"] = objects["panels"].eq(len(panels))
    objects["unchanged"] = objects["paired"] & objects["predictions"].eq(1) & objects["valid"]
    rows = []
    for label in labels:
        subset = objects[objects["true_label"].eq(label) & objects["paired"]]
        minimum = recall[label].min()
        maximum = recall[label].max()
        rows.append({"label": label, "min_support": int(object_support[label].min()),
                     "observations": int(totals[label].sum()), "min_recall": minimum, "max_recall": maximum,
                     "recall_range": maximum - minimum, "mean_recall": recall[label].mean(),
                     "unchanged_fraction": float(subset["unchanged"].mean()) if len(subset) and len(panels) >= 2 else np.nan,
                     "paired_objects": len(subset), "panels": len(panels)})
    return pd.DataFrame(rows, columns=columns)


def confusion_clustering(data: pd.DataFrame, labels: list[str], clusters: int) -> tuple[list[str], dict[str, str]]:
    from scipy.cluster.hierarchy import cut_tree, leaves_list, linkage
    from scipy.spatial.distance import squareform

    labels = sorted(labels)
    if len(labels) < 2:
        return labels, {label: "Cluster 1" for label in labels}
    affinity = np.zeros((len(labels), len(labels)))
    panels = list(data.groupby("panel_id", sort=True))
    for _, panel in panels:
        matrix = pd.crosstab(panel["true_label"], panel["prediction"]).reindex(index=labels, columns=labels, fill_value=0).astype(float)
        # Denominators include failed responses; each reference panel has equal weight.
        totals = panel.groupby("true_label").size().reindex(labels, fill_value=0)
        rates = matrix.div(totals.replace(0, np.nan), axis=0).fillna(0.).to_numpy(copy=True)
        np.fill_diagonal(rates, 0.)
        affinity += (rates + rates.T) / 2.
    affinity /= max(1, len(panels))
    distance = np.clip(1. - affinity, 0., 1.)
    np.fill_diagonal(distance, 0.)
    tree = linkage(squareform(distance, checks=True), method="average", optimal_ordering=True)
    ordered = [labels[int(index)] for index in leaves_list(tree)]
    assignment = cut_tree(tree, n_clusters=[min(len(labels), max(1, clusters))]).reshape(-1)
    raw = dict(zip(labels, assignment.tolist()))
    names: dict[int, str] = {}
    for label in ordered:
        names.setdefault(raw[label], f"Cluster {len(names) + 1}")
    return ordered, {label: names[raw[label]] for label in labels}


def semantic_groups(labels: list[str], dataset: str) -> dict[str, str]:
    groups = ({
        "Structure": "beam,column,wall", "Surfaces": "floor,ceiling", "Openings": "door,window",
        "Furniture": "chair,table,bookcase,sofa", "Other indoor objects": "board,clutter,stairs",
    } if dataset == "s3dis" else {
        "Seating": "chair,bench,stool,sofa", "Tables": "table,desk",
        "Storage": "bookshelf,wardrobe,dresser,tv_stand,night_stand",
        "Sanitary": "sink,toilet,bathtub", "Vessels and plants": "bottle,bowl,cup,vase,flower_pot,plant",
        "Electronics": "laptop,keyboard,monitor,radio,xbox", "Vehicles": "airplane,car",
    })
    lookup = {label: group for group, members in groups.items() for label in members.split(",")}
    return {label: lookup.get(label, label) for label in labels}


def make_reference(data: pd.DataFrame, settings: MatrixSettings, manual_labels: list[str] | None = None,
                   custom_groups: dict[str, str] | None = None) -> dict[str, Any]:
    profiles = class_profiles(data)
    labels = sorted(profiles["label"].tolist())
    needs_clusters = settings.order == "cluster" or settings.grouping == "cluster"
    clustered, cluster_groups = (confusion_clustering(data, labels, settings.clusters) if needs_clusters
                                else (labels, {label: label for label in labels}))
    stable = (profiles["min_support"].ge(settings.min_support)
              & profiles["unchanged_fraction"].ge(settings.stable_fraction)
              & profiles["recall_range"].le(settings.recall_tolerance) & profiles["panels"].ge(2))
    profiles["stable"] = stable
    profiles["stable_good"] = stable & profiles["min_recall"].ge(settings.good_recall)
    profiles["stable_bad"] = stable & profiles["max_recall"].le(settings.bad_recall)
    eligible = profiles[profiles["min_support"].ge(settings.min_support)]
    if settings.focus == "errors":
        shown = eligible.sort_values(["min_recall", "label"])["label"].head(settings.top_k).tolist()
    elif settings.focus == "changed":
        shown = eligible.sort_values(["recall_range", "label"], ascending=[False, True])["label"].head(settings.top_k).tolist()
    elif settings.focus == "compact":
        shown = profiles.loc[~stable, "label"].tolist()
    elif settings.focus in {"stable_good", "stable_bad"}:
        shown = profiles.loc[profiles[settings.focus], "label"].tolist()
    elif settings.focus == "manual":
        shown = [label for label in (manual_labels or []) if label in labels]
    else:
        shown = labels
    if settings.order == "cluster":
        order = clustered
    elif settings.order == "recall":
        order = profiles.sort_values(["mean_recall", "label"], na_position="last")["label"].tolist()
    else:
        order = labels
    dataset = str(data.iloc[0]["dataset"]) if not data.empty else "modelnet"
    mapping = ({label: label for label in labels} if settings.grouping == "none" else
               cluster_groups if settings.grouping == "cluster" else
               (custom_groups or semantic_groups(labels, dataset)) if settings.grouping == "semantic" else
               {label: (custom_groups or {}).get(label, label).strip() or label for label in labels})
    mapping = {label: str(mapping.get(label, label)).strip() or label for label in labels}
    reference = {"schema_version": 1, "dataset": dataset, "order": order, "shown": shown,
            "groups": mapping, "settings": settings.public(),
            "cluster_method": "average linkage on 1 - mean symmetric off-diagonal row shares" if needs_clusters else "none"}
    reference["count_max"] = max([int(matrix_values(panel, reference, settings)["counts"].to_numpy().max(initial=0))
                                  for _, panel in data.groupby("panel_id", sort=True)] + [1])
    return reference


def matrix_values(panel: pd.DataFrame, reference: dict[str, Any], settings: MatrixSettings) -> dict[str, Any]:
    order = list(reference["order"])
    shown = set(reference["shown"])
    mapping = {label: reference["groups"].get(label, label) if label in shown else OTHER for label in order}
    groups = list(dict.fromkeys(mapping[label] for label in order if mapping[label] != OTHER))
    if any(label not in shown for label in order):
        groups.append(OTHER)
    columns = [*groups, *([INVALID] if settings.include_invalid else [])]
    data = panel if settings.include_invalid else panel[panel["prediction"].ne(INVALID)]
    counts = pd.DataFrame(0, index=groups, columns=columns, dtype=int)
    correct = counts.copy()
    for (truth, predicted), subset in data.groupby(["true_label", "prediction"], sort=False):
        row = mapping.get(truth, OTHER if OTHER in groups else None)
        column = INVALID if predicted == INVALID else mapping.get(predicted, OTHER if OTHER in columns else None)
        if row is None or column not in columns:
            continue
        counts.loc[row, column] += len(subset)
        if truth == predicted:
            correct.loc[row, column] += len(subset)
    support = counts.sum(axis=1)
    visible_counts = counts - correct if settings.errors_only else counts.copy()
    values = visible_counts.astype(float)
    if settings.normalize:
        values = values.div(support.replace(0, np.nan), axis=0)
    else:
        values.loc[support.eq(0)] = np.nan
    # Mask for display only; exported counts and denominators remain intact.
    displayed = values.mask(values.lt(settings.mask_below))
    names = {group: reference.get("other_name", f"Other ({sum(label not in shown for label in order)} classes)") if group == OTHER else group for group in groups}
    return {"counts": counts, "correct": correct, "support": support, "values": values,
            "displayed": displayed, "names": names, "members": mapping}


def matrix_export(data: pd.DataFrame, reference: dict[str, Any], settings: MatrixSettings) -> pd.DataFrame:
    rows = []
    for panel_id, panel in data.groupby("panel_id", sort=True):
        matrix = matrix_values(panel, reference, settings)
        identity = {key: panel.iloc[0][key] for key in [*SERIES, "condition", "view_count"]}
        for truth in matrix["counts"].index:
            for prediction in matrix["counts"].columns:
                count = int(matrix["counts"].loc[truth, prediction])
                correct = int(matrix["correct"].loc[truth, prediction])
                rows.append({**identity, "panel_id": panel_id, "true_group": truth, "predicted_group": prediction,
                             "count": count, "original_correct": correct, "original_errors": count - correct,
                             "row_support": int(matrix["support"].loc[truth]), "display_value": matrix["values"].loc[truth, prediction]})
    return pd.DataFrame(rows)


def condition_sort(condition: str) -> tuple[int, str]:
    if condition.startswith("views_"):
        try:
            return int(condition[6:]), condition
        except ValueError:
            pass
    return {"rgb": 1, "rgb_aabb": 2}.get(condition, 10), condition


def paired_changes(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for series_id, series in data.groupby("series_id", sort=True):
        conditions = sorted(series["condition"].unique(), key=condition_sort)
        if len(conditions) < 2:
            continue
        before, after = conditions[0], conditions[-1]
        # Compare matching objects AND repetitions. Ambiguous duplicate observations are excluded.
        keys = ["sample_id", "repeat", "true_label"]
        left = series[series["condition"].eq(before)].drop_duplicates(keys, keep=False)
        right = series[series["condition"].eq(after)].drop_duplicates(keys, keep=False)
        pairs = left[keys + ["is_correct"]].merge(right[keys + ["is_correct"]], on=keys, suffixes=("_before", "_after"))
        for label, group in pairs.groupby("true_label", sort=True):
            gain = (~group["is_correct_before"] & group["is_correct_after"]).sum()
            loss = (group["is_correct_before"] & ~group["is_correct_after"]).sum()
            rows.append({"series_id": series_id, "label": label, "before": before, "after": after,
                         "paired_observations": len(group), "gains": int(gain), "losses": int(loss),
                         "delta_recall": float((gain - loss) / len(group))})
    return pd.DataFrame(rows)


def confusion_figure(data: pd.DataFrame, reference: dict[str, Any], settings: MatrixSettings,
                     *, title: str = "Confusion matrices", expected_conditions: list[str] | None = None) -> go.Figure:
    series = sorted(data["series_id"].unique()) if not data.empty else []
    conditions = expected_conditions or sorted(data["condition"].unique(), key=condition_sort)
    if not series or not conditions:
        figure = go.Figure()
        figure.add_annotation(text="No observations for this selection", showarrow=False)
        figure.update_layout(template="simple_white", height=320)
        return figure
    titles = []
    for identity in series:
        for condition in conditions:
            text = condition.replace("views_", "Views: ").replace("rgb_aabb", "RGB + AABB").replace("rgb", "RGB")
            titles.append(text)
    n = max(1, len(matrix_values(data.iloc[:0], reference, settings)["counts"].index))
    figure_height = max(580, (max(280, 15 * n) + 190) * len(series) + 140)
    vertical_gap = min(180. / (figure_height - 310), .9 / max(1, len(series) - 1)) if len(series) > 1 else 0.
    figure = make_subplots(rows=len(series), cols=len(conditions), subplot_titles=titles, shared_yaxes=True,
                          horizontal_spacing=min(.045, .2 / len(conditions)), vertical_spacing=vertical_gap)
    matrices = {}
    for identity in series:
        for condition in conditions:
            panel = data[data["series_id"].eq(identity) & data["condition"].eq(condition)]
            if not panel.empty:
                matrices[(identity, condition)] = matrix_values(panel, reference, settings)
    maximum = 1. if settings.normalize else reference.get("count_max", max([int(item["counts"].to_numpy().max(initial=0)) for item in matrices.values()] + [1]))
    for row_index, identity in enumerate(series, 1):
        label = identity.replace("current_paper_protocol | ", "Current | ").replace("modelnet33_exact_ids | ", "Legacy aligned | ")
        figure.add_annotation(text=f"<b>{label}</b>", x=0., y=1., xref=f"x{'' if row_index == 1 else (row_index-1)*len(conditions)+1} domain",
                              yref=f"y{'' if row_index == 1 else (row_index-1)*len(conditions)+1} domain", yshift=56, xanchor="left", showarrow=False, font={"size": 12})
        for column_index, condition in enumerate(conditions, 1):
            matrix = matrices.get((identity, condition))
            axis = (row_index - 1) * len(conditions) + column_index
            suffix = "" if axis == 1 else str(axis)
            if matrix is None:
                figure.add_annotation(text="Not evaluated", x=.5, y=.5, xref=f"x{suffix} domain", yref=f"y{suffix} domain", showarrow=False)
                continue
            support = matrix["support"].to_numpy()
            custom = np.stack([matrix["counts"].to_numpy(), matrix["correct"].to_numpy(),
                               matrix["counts"].to_numpy() - matrix["correct"].to_numpy(),
                               np.repeat(support[:, None], len(matrix["counts"].columns), axis=1)], axis=-1)
            figure.add_trace(go.Heatmap(
                z=matrix["displayed"].to_numpy(), x=[matrix["names"].get(value, value) for value in matrix["counts"].columns],
                y=[matrix["names"][value] for value in matrix["counts"].index], customdata=custom, coloraxis="coloraxis",
                hovertemplate="True group: %{y}<br>Predicted group: %{x}<br>Displayed: %{z:.3f}<br>Count: %{customdata[0]}<br>Originally correct: %{customdata[1]}<br>Originally wrong/invalid: %{customdata[2]}<br>Row denominator: %{customdata[3]}<extra></extra>",
                hoverongaps=False,
            ), row=row_index, col=column_index)
    figure.update_layout(title={"text": title, "x": .02}, template="simple_white", font={"family": "Arial", "size": 12},
                         width=max(750, 370 * len(conditions)), height=figure_height,
                         margin={"l": 120, "r": 95, "t": 130, "b": 180},
                         coloraxis={"colorscale": "Blues", "cmin": 0, "cmax": maximum,
                                    "colorbar": {"title": "Row share" if settings.normalize else "Count", "tickformat": ".0%" if settings.normalize else "d", "thickness": 16}},
                         paper_bgcolor="white", plot_bgcolor="#eeeeee")
    note = ("Row shares include invalid responses." if settings.include_invalid else "Row shares conditional on valid responses.") if settings.normalize else "Absolute observation counts; common reference color scale."
    if settings.grouping != "none" or len(reference["shown"]) != len(reference["order"]):
        note += "<br>Grouped diagonals include within-group class errors; they are not class accuracy."
    if settings.errors_only:
        note += "<br>Only original class errors shown; denominators unchanged."
    figure.add_annotation(text=note, x=0., y=0., xref="paper", yref="paper", yshift=-142,
                          xanchor="left", yanchor="top", align="left", showarrow=False, font={"size": 10})
    for row in range(1, len(series)+1):
        for col in range(1, len(conditions)+1):
            figure.update_xaxes(title_text="Predicted class / group", tickangle=55, tickfont={"size": 10}, showgrid=False, row=row, col=col)
            figure.update_yaxes(title_text="True class / group" if col == 1 else "", showticklabels=col == 1,
                                autorange="reversed", tickfont={"size": 10}, showgrid=False, row=row, col=col)
    return figure
