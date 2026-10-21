"""Small, deterministic summaries on a shared object cohort within one protocol."""
from __future__ import annotations

import pandas as pd

from .confusion_analysis import INVALID, class_profiles, paired_changes


def automatic_digest(data: pd.DataFrame, min_support: int = 5) -> dict:
    if data.empty:
        return {}
    if data["protocol_scope"].nunique() != 1 or data["dataset"].nunique() != 1:
        raise ValueError("Die automatische Auswertung benötigt einen Datensatz und ein Protokoll.")
    sizes = data.groupby(["series_id", "panel_id"])["sample_id"].nunique()
    minimum = sizes.groupby("series_id").min()
    selected = minimum[minimum.ge(minimum.max() * .95)].index
    chosen = data[data["series_id"].isin(selected)].copy()
    # Compare the same objects; retain failures as incorrect observations.
    presence = chosen.groupby("sample_id")["panel_id"].nunique()
    common = presence[presence.eq(chosen["panel_id"].nunique())].index
    chosen = chosen[chosen["sample_id"].isin(common)].copy()
    if chosen.empty:
        return {"empty_cohort": True}
    chosen["model_label"] = chosen["provider"] + " / " + chosen["model"]
    summary = chosen.groupby(["series_id", "model_label", "condition", "view_count"], dropna=False).agg(
        accuracy=("is_correct", "mean"), objects=("sample_id", "nunique"),
        observations=("sample_id", "size"), coverage=("prediction", lambda x: x.ne(INVALID).mean()),
    ).reset_index()
    if chosen["dataset"].iloc[0] == "modelnet":
        final = chosen[chosen["view_count"].eq(chosen["view_count"].max())]
    else:
        last = "rgb_aabb" if chosen["condition"].eq("rgb_aabb").any() else sorted(chosen["condition"].unique())[-1]
        final = chosen[chosen["condition"].eq(last)]
    ranking = final.groupby(["series_id", "model_label"]).agg(
        accuracy=("is_correct", "mean"), objects=("sample_id", "nunique"),
        observations=("sample_id", "size"), coverage=("prediction", lambda x: x.ne(INVALID).mean()),
    ).reset_index().sort_values(["accuracy", "model_label"], ascending=[False, True])
    best = ranking.iloc[0]
    detail = final[final["series_id"].eq(best["series_id"])]
    support = detail.groupby("true_label").size()
    pairs = detail[~detail["is_correct"]].groupby(["true_label", "prediction"]).size().rename("errors").reset_index()
    pairs["support"] = pairs["true_label"].map(support)
    pairs["share"] = pairs["errors"] / pairs["support"]
    pairs = pairs[pairs["support"].ge(min_support)].sort_values(["errors", "true_label", "prediction"], ascending=[False, True, True]).head(5)
    profiles = class_profiles(chosen)
    enough = profiles[profiles["min_support"].ge(min_support)]
    difficult = enough[enough["max_recall"].lt(.5)].sort_values(["mean_recall", "label"]).head(5)
    stable = enough[enough["panels"].ge(2) & enough["min_recall"].ge(.8)
                    & enough["recall_range"].le(.05) & enough["unchanged_fraction"].ge(.95)]
    changes = paired_changes(chosen)
    gains = pd.DataFrame()
    if not changes.empty:
        gains = changes.groupby(["series_id", "before", "after"]).agg(
            paired_observations=("paired_observations", "sum"), gains=("gains", "sum"), losses=("losses", "sum"),
        ).reset_index()
        gains["delta"] = (gains["gains"] - gains["losses"]) / gains["paired_observations"]
        gains["model_label"] = gains["series_id"].map(chosen.set_index("series_id")["model_label"].to_dict())
        gains = gains.sort_values(["delta", "model_label"], ascending=[False, True])
    return dict(data=chosen, summary=summary, ranking=ranking, detail=detail, pairs=pairs,
                difficult=difficult, stable=stable, gains=gains, profiles=profiles,
                excluded_series=int(data["series_id"].nunique() - chosen["series_id"].nunique()),
                excluded_observations=int(len(data) - len(chosen)), condition=final["condition"].iloc[0])
