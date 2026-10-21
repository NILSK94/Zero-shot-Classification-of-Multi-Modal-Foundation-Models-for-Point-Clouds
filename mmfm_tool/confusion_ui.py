from __future__ import annotations

import json
import io
import zipfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from .confusion_analysis import (
    MatrixSettings, class_profiles, confusion_figure, make_reference, matrix_export,
    paired_changes, prepare_confusion_rows, semantic_groups,
)
from .run_comparison import merge_run_results, run_scope
from .historical_csv import load_historical_csvs
from .historical_overview import historical_overview_figure, overview_panels, overview_values, save_historical_overview
from .result_digest import automatic_digest

FOCUS = {"all": 'All classes', "errors": 'Most difficult classes (top N)',
         "changed": 'Largest changes in class recognition (top N)', "compact": 'Combine stable classes',
         "stable_good": 'Consistently well-recognised classes', "stable_bad": 'Consistently difficult classes',
         "manual": 'Select classes manually'}
ORDER = {"alphabetical": 'Alphabetical', "recall": 'Difficult classes first', "cluster": 'Cluster by confusions'}
GROUPING = {"none": 'Individual classes', "semantic": 'Suggested semantic groups', "cluster": 'Confusion clusters', "custom": 'Custom groups'}


@st.cache_data(show_spinner=False)
def _load_runs(paths: tuple[str, ...], signatures: tuple[tuple[int, int], ...]) -> pd.DataFrame:
    del signatures
    return merge_run_results(paths)


@st.cache_data(show_spinner=False)
def _profiles(data: pd.DataFrame) -> pd.DataFrame:
    return class_profiles(data)


def _validate_preset(preset: dict) -> MatrixSettings:
    if preset.get("schema_version") != 1 or not isinstance(preset.get("dataset"), str):
        raise ValueError("Unknown reference format.")
    order, shown, groups = preset["order"], preset["shown"], preset["groups"]
    if not isinstance(order, list) or not 0 < len(order) <= 200 or not all(isinstance(label, str) for label in order):
        raise ValueError('Invalid class order.')
    if len(set(order)) != len(order) or not isinstance(shown, list) or not set(shown).issubset(order):
        raise ValueError('Invalid focus classes.')
    if not isinstance(groups, dict) or set(groups) != set(order) or not all(isinstance(value, str) and value.strip() for value in groups.values()):
        raise ValueError('Invalid groups.')
    settings = MatrixSettings(**preset["settings"])
    if settings.focus not in FOCUS or settings.order not in ORDER or settings.grouping not in GROUPING:
        raise ValueError("Unknown display option.")
    for name in ("stable_fraction", "recall_tolerance", "good_recall", "bad_recall"):
        if not isinstance(getattr(settings, name), (int, float)) or not 0 <= getattr(settings, name) <= 1:
            raise ValueError('Thresholds must lie between 0 and 1.')
    if not isinstance(settings.mask_below, (float, int)) or not 0 <= settings.mask_below <= (1 if settings.normalize else 100000):
        raise ValueError('Invalid cell mask.')
    if not 1 <= settings.top_k <= 100 or not 1 <= settings.min_support <= 100000 or not 1 <= settings.clusters <= 40:
        raise ValueError('Invalid number of classes, groups or observations.')
    for name in ("normalize", "errors_only", "include_invalid"):
        if not isinstance(getattr(settings, name), bool):
            raise ValueError('Invalid display option.')
    return settings


@st.cache_data(show_spinner=False)
def _load_archive(path: str, signatures: tuple[tuple[str, int, int], ...], original_rows: bool = False):
    del signatures
    return load_historical_csvs(path, original_rows=original_rows)


def _historical_overview(archive: pd.DataFrame, settings: MatrixSettings, output_root: Path, *, use_saved: bool = True) -> None:
    st.subheader('Historical overview in the original layout')
    real_model = st.selectbox('BLK2GO model in the bottom row', ["gemini-1.5-pro", "gpt-4o"], key="history_real_model")
    one, two = st.columns(2)
    show_values = one.checkbox('Show cell values in the historical overview', value=True, key="history_values")
    show_zeros = two.checkbox('Label zero values too', value=False, key="history_zeros")
    saved = st.session_state.get("confusion_display_settings", {}) if use_saved else {}
    references = {dataset: (item[0], item[1]) for dataset, item in saved.items() if dataset in {"modelnet", "blk2go"}}
    panels, references = overview_panels(archive, settings, references, real_model)
    figure = historical_overview_figure(panels, references, show_values=show_values, show_zeros=show_zeros)
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False,
        "toImageButtonOptions": {"format": "svg", "filename": "historical_overview", "scale": 3}})
    table = pd.DataFrame([{**{key: panel.get(key) for key in ["row", "col", "dataset", "model", "scope", "condition", "objects", "observations"]},
                           "state": "loaded" if "matrix" in panel else "not reconstructed"} for panel in panels])
    with st.expander('Historical panels and source data'):
        st.caption('ModelNet follows the original script: each CSV row counts, and the class is taken from the first filename segment. Multiple views are not independent objects. Summaries use object-level decisions. Pilot, full-test and BLK2GO results stay separate, with shared scales within each dataset. Group diagonals are not class accuracy. Stable Diffusion remains a placeholder without unambiguous class predictions.')
        st.dataframe(table, hide_index=True, width="stretch")
    st.download_button('Download historical overview as interactive HTML', figure.to_html(include_plotlyjs=True).encode("utf-8"),
                        file_name="historical_overview.html", mime="text/html", key="history_html")
    st.download_button('Download historical matrix values as CSV', overview_values(panels).to_csv(index=False).encode("utf-8"),
                        file_name="historical_overview.csv", mime="text/csv", key="history_csv")
    if st.button('Generate historical SVG and PNG package', key="history_export"):
        with st.spinner('Generating the historical overview as a vector graphic and high-resolution image …'):
            try:
                paths = save_historical_overview(output_root, panels, references, figure,
                                                  show_values=show_values, show_zeros=show_zeros)
                output = io.BytesIO()
                with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                    for path in paths.values():
                        bundle.write(path, arcname=path.name)
                st.session_state["history_bundle"] = output.getvalue()
                st.session_state["history_bundle_source"] = str(output_root)
                st.success(f"SVG, PNG, HTML, CSV and reference JSON saved: {output_root}")
            except (ImportError, OSError, ValueError) as exc:
                st.error(f"Could not generate the graphics package: {exc}")
    if "history_bundle" in st.session_state:
        st.download_button('Download the latest historical graphics package', st.session_state["history_bundle"],
                           file_name="historical_overview.zip", mime="application/zip", key="history_zip")
        st.caption('The ZIP contains the last generated version. Regenerate the package after changing filters.')


def _render_expert(run_dirs: list[Path], historical_root: Path | None = None,
                             overview_output_root: Path | None = None) -> None:
    st.subheader("Shared confusion matrix explorer")
    st.caption('The same filters, class order and colour scales apply to historical and current results. ModelNet and S3DIS are evaluated separately.')
    if not st.checkbox('Enable matrix explorer', value=False, key="cm_enabled"):
        return
    default = [path for path in run_dirs if run_scope(path) != "modelnet40_legacy_full"]
    selected = st.multiselect("Matrix explorer: runs", run_dirs, default=default, format_func=lambda path: path.name,
                             key="cm_runs", help='Imported historical results can also be selected. The full ModelNet40 audit is available separately.')
    signatures = tuple(((path / "results.csv").stat().st_mtime_ns, (path / "results.csv").stat().st_size) for path in selected)
    with st.spinner('Loading results for shared matrix filters …'):
        merged = _load_runs(tuple(str(path) for path in selected), signatures)
    archive_root = Path(st.text_input('Historical CSV folder', str(historical_root or Path("data_results_csv")), key="cm_archive_path"))
    archive_enabled = st.checkbox('Include historical CSV files', value=archive_root.is_dir(), key="cm_archive_enabled")
    archive = pd.DataFrame()
    if archive_enabled:
        archive_files = sorted(archive_root.glob("*.csv"))
        archive, audit = _load_archive(str(archive_root), tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size) for path in archive_files))
        with st.expander('Historical source files and reconstruction status'):
            st.dataframe(audit, hide_index=True, width="stretch")
        if not archive.empty:
            merged = pd.concat([merged, archive], ignore_index=True, sort=False)
    historical_layout = False
    if not archive.empty:
        historical_layout = st.radio("Matrix layout", ['Selected models and conditions', 'Historical overview (original layout)'],
                                     key="cm_layout", horizontal=True) == 'Historical overview (original layout)'
        if historical_layout:
            archive, audit = _load_archive(str(archive_root), tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size)
                                         for path in sorted(archive_root.glob("*.csv"))), original_rows=True)
            merged = archive.copy()
            _historical_overview(archive, MatrixSettings(normalize=False),
                                 overview_output_root or Path.cwd() / "artifacts/historical_overview", use_saved=False)
            return
    if merged.empty:
        st.info('These runs contain no result rows.')
        return
    dataset = st.selectbox('Matrix explorer: dataset', sorted(merged["dataset"].astype(str).unique()), key="cm_dataset")
    candidates = prepare_confusion_rows(merged, dataset, paired=False)
    options = sorted(candidates["series_id"].unique())
    panel_sizes = candidates.groupby(["series_id", "panel_id"])["sample_id"].nunique()
    minimum_sizes = panel_sizes.groupby("series_id").min()
    default_series = options if historical_layout else minimum_sizes[minimum_sizes.ge(minimum_sizes.max() * .95)].index.tolist()
    reference_series = st.multiselect('Models and protocols used as the reference', options, default=default_series,
                                     key="cm_reference_series", help='Complete series are selected by default. Pilot and partial runs can be added; their shared object IDs may reduce the comparison cohort.')
    candidates = candidates[candidates["series_id"].isin(reference_series)]
    if candidates.empty:
        st.info('Select at least one model/protocol series as the reference.')
        return
    paired = (False if historical_layout else st.checkbox('Use only object IDs shared by the selected panels', value=True, key="cm_paired",
                         help='API errors remain included. Uses the same objects to compare available model/protocol/condition panels.'))
    data = prepare_confusion_rows(candidates, dataset, paired=paired)
    if data.empty:
        st.warning('No shared object IDs. Change the selected runs or disable the shared-object filter.')
        return
    st.caption(f"Reference cohort: {data['sample_id'].nunique():,} object IDs · {len(data):,} observations · {data['panel_id'].nunique()} panels. Missing conditions are not shown as zero results.")
    with st.expander('Shared filters, clusters and reference', expanded=False):
        uploaded = st.file_uploader('Load a saved matrix reference', type=["json"], key="cm_upload")
        if uploaded is not None and st.button("Apply matrix reference", key="cm_apply"):
            try:
                preset = json.loads(uploaded.getvalue())
                if "references" in preset:
                    historical_presets = preset["references"]
                    preset = historical_presets[dataset]
                    for other_dataset, other_reference in historical_presets.items():
                        if other_dataset != dataset:
                            other_settings = _validate_preset(other_reference)
                            st.session_state[f"cm_reference_{other_dataset}"] = other_reference
                            st.session_state[f"cm_signature_{other_dataset}"] = None
                            st.session_state.setdefault("confusion_display_settings", {})[other_dataset] = (other_settings, other_reference, pd.DataFrame())
                loaded_settings = _validate_preset(preset)
                if preset["dataset"] != dataset:
                    raise ValueError('The saved reference belongs to a different dataset.')
                for name, value in loaded_settings.public().items():
                    if name == "mask_below" and not loaded_settings.normalize:
                        continue
                    st.session_state[f"cm_{name}"] = value
                if not loaded_settings.normalize:
                    st.session_state["cm_count_mask"] = int(loaded_settings.mask_below)
                st.session_state[f"cm_reference_{dataset}"] = preset
                st.session_state["cm_manual"] = preset["shown"]
                st.session_state["cm_custom_seed"] = preset["groups"]
                st.session_state.pop("cm_group_editor", None)
                st.session_state[f"cm_signature_{dataset}"] = None
                st.success('Reference loaded. Order and focus remain shared across the selected models.')
            except (ValueError, TypeError, KeyError) as exc:
                st.error(f"Could not load the reference: {exc}")
        first, second, third = st.columns(3)
        focus = first.selectbox("Class focus", list(FOCUS), format_func=FOCUS.get, key="cm_focus")
        top_k = int(second.number_input('Top N classes', min_value=1, max_value=100, value=12, key="cm_top_k"))
        min_support = int(third.number_input('Minimum samples per class and panel', min_value=1, max_value=100000, value=5, key="cm_min_support"))
        first, second = st.columns(2)
        order = first.selectbox('Shared class order', list(ORDER), format_func=ORDER.get, key="cm_order")
        grouping = second.selectbox('Group classes', list(GROUPING), format_func=GROUPING.get, key="cm_grouping")
        clusters = int(st.number_input('Number of confusion clusters', min_value=1, max_value=40, value=6, key="cm_clusters"))
        labels = sorted(set(data["true_label"]) | (set(data["prediction"]) - {"[No valid response]"}))
        reference_saved = st.session_state.get(f"cm_reference_{dataset}", {})
        options = sorted(set(labels) | set(reference_saved.get("order", [])))
        manual = st.multiselect("Custom focus classes", options, default=options, key="cm_manual") if focus == "manual" else []
        custom = None
        if grouping in {"custom", "semantic"}:
            seed = st.session_state.get("cm_custom_seed", semantic_groups(options, dataset))
            edited = st.data_editor(pd.DataFrame([{"Class": label, "Group": seed.get(label, label)} for label in options]),
                                    disabled=["Class"], hide_index=True, key="cm_group_editor")
            custom = dict(zip(edited["Class"], edited["Group"].fillna("").astype(str)))
            if grouping == "semantic":
                # Semantic suggestions are editable; use the edited mapping for the actual result.
                st.caption('These are exploratory suggestions, not a validated taxonomy. Group names can be edited here.')
        first, second = st.columns(2)
        stable_fraction = first.slider('Minimum fraction of unchanged object predictions', 0., 1., .95, .01, key="cm_stable_fraction")
        recall_tolerance = second.slider('Maximum recognition-rate range for stable classes', 0., 1., .05, .01, key="cm_recall_tolerance")
        first, second = st.columns(2)
        good_recall = first.slider('Consistently well-recognised: minimum recognition rate', 0., 1., .8, .05, key="cm_good_recall")
        bad_recall = second.slider('Consistently difficult: maximum recognition rate', 0., 1., .2, .05, key="cm_bad_recall")
        first, second, third = st.columns(3)
        normalize = first.checkbox("Row shares instead of counts", value=True, key="cm_normalize")
        errors_only = second.checkbox('Show errors only', value=False, key="cm_errors_only")
        include_invalid = third.checkbox('Show missing or invalid responses', value=True, key="cm_include_invalid")
        mask_below = (st.slider("Hide cells below this share", 0., 1., 0., .01, key="cm_mask_below",
                               help='Display mask only; values, denominators and exported counts remain unchanged.') if normalize else
                      float(st.number_input('Hide cells below this count', min_value=0, max_value=100000, value=0, key="cm_count_mask")))
        lock = st.checkbox('Keep the reference fixed when switching runs or models', value=True, key="cm_lock")
        refresh = st.button('Recalculate focus and clusters from this reference', key="cm_refresh")
    settings = MatrixSettings(focus, top_k, min_support, stable_fraction, recall_tolerance, good_recall,
                              bad_recall, order, grouping, clusters, normalize, errors_only, mask_below, include_invalid)
    signature = json.dumps({"settings": settings.public(), "manual": manual, "custom": custom}, sort_keys=True)
    reference_key, signature_key = f"cm_reference_{dataset}", f"cm_signature_{dataset}"
    saved_signature = st.session_state.get(signature_key)
    if (reference_key not in st.session_state or not lock or refresh
            or (saved_signature is not None and saved_signature != signature)):
        with st.spinner('Calculating class profiles and shared clusters …'):
            reference = make_reference(data, settings, manual, custom)
        st.session_state[reference_key] = reference
    reference = st.session_state[reference_key]
    st.session_state[signature_key] = signature
    missing_labels = set(labels) - set(reference["order"])
    if missing_labels:
        st.warning('New classes are missing from the fixed reference: ' + ", ".join(sorted(missing_labels)) + '. Please recalculate the reference.')
        return
    st.session_state.setdefault("confusion_display_settings", {})[dataset] = (settings, reference, data)
    st.caption('Focus and order are derived from the shared reference, rather than recalculated for each model. Changing filters recalculates the focus; switching the displayed model does not.')
    if historical_layout:
        _historical_overview(archive, settings, overview_output_root or Path.cwd() / "artifacts" / "historical_overview")
        return
    series_options = sorted(data["series_id"].unique())
    shown_series = st.multiselect('Displayed models and protocols', series_options,
                                  default=series_options[:2], key="cm_series")
    displayed = data[data["series_id"].isin(shown_series)]
    if displayed.empty:
        st.info('Select at least one model/protocol to display.')
        return
    totals = st.columns(3)
    totals[0].metric('Accuracy – full selection', f"{displayed['is_correct'].mean():.1%}")
    totals[1].metric('Coverage – full selection', f"{displayed['prediction'].ne('[No valid response]').mean():.1%}")
    totals[2].metric('Classes shown individually', f"{len(reference['shown'])}/{len(reference['order'])}")
    st.caption('The metrics above use all observations in the displayed panels before class filtering. With multiple models they are weighted by observation count; individual metrics remain available in the run evaluation.')
    if settings.grouping != "none" or len(reference["shown"]) != len(reference["order"]):
        st.info('Groups and Other combine original classes. Their diagonals include within-group errors and do not represent class accuracy. Hover details and CSV files distinguish originally correct and incorrect responses.')
    if not settings.include_invalid:
        st.caption('The matrix shows valid responses only; row shares are conditional on valid responses. Accuracy and coverage above still use the full selection.')
    figure = confusion_figure(displayed, reference, settings, title=f"{dataset.upper()} · shared class reference")
    export_format = st.selectbox("Grafik-Downloadformat", ["svg", "png"], key="cm_export_format",
                                 help='Use the chart camera button to export vector SVG or high-resolution PNG.')
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "toImageButtonOptions": {
        "format": export_format, "filename": f"{dataset}_confusion_filtered", "scale": 3}})
    st.caption('Grey cells indicate missing observations or masked values. White indicates an observed zero. Use the chart camera button to export SVG or PNG.')
    with st.expander('Class profiles, stable placeholders and group assignments'):
        profiles = _profiles(data)
        profiles["shown"] = profiles["label"].isin(reference["shown"])
        profiles["group"] = profiles["label"].map(reference["groups"])
        st.dataframe(profiles, hide_index=True, width="stretch")
        st.caption('Stability requires at least two panels, sufficient samples in each panel, a small recognition-rate range, and unchanged predictions for the same objects. Rare classes are not automatically hidden as stable.')
    changes = paired_changes(displayed)
    if not changes.empty:
        with st.expander('Gains and losses on the same objects', expanded=False):
            st.caption('ModelNet: first versus last available view count. S3DIS: RGB versus RGB+AABB. Only unambiguous object/repetition pairs are compared; no significance claim is made.')
            chart = px.bar(changes, x="label", y="delta_recall", color="series_id", barmode="group",
                           hover_data=["paired_observations", "gains", "losses", "before", "after"],
                           labels={"label": "Class", "delta_recall": "Class recognition change", "series_id": "Model / protocol"})
            chart.update_layout(template="simple_white")
            chart.update_yaxes(range=[-1., 1.], tickformat=".0%")
            st.plotly_chart(chart, width="stretch")
            st.dataframe(changes, hide_index=True, width="stretch")
    exported = matrix_export(displayed, reference, settings)
    preset = {**reference, "settings": settings.public(), "source_runs": [str(path) for path in selected],
              "source_signatures": signatures, "paired_objects": paired,
              "historical_csv_root": str(archive_root) if archive_enabled else None,
              "reference_series": reference_series,
              "historical_csv_signatures": [(path.name, path.stat().st_mtime_ns, path.stat().st_size)
                                             for path in sorted(archive_root.glob("*.csv"))] if archive_enabled else []}
    one, two, three = st.columns(3)
    one.download_button('Download matrix as interactive HTML', figure.to_html(include_plotlyjs=True).encode("utf-8"),
                        file_name=f"{dataset}_confusion.html", mime="text/html", key="cm_html")
    two.download_button('Download matrix values and original errors as CSV', exported.to_csv(index=False).encode("utf-8"),
                        file_name=f"{dataset}_confusion.csv", mime="text/csv", key="cm_csv")
    three.download_button('Save filters and reference', json.dumps(preset, ensure_ascii=False, indent=2).encode("utf-8"),
                          file_name=f"{dataset}_confusion_reference.json", mime="application/json", key="cm_json")


def render_confusion_explorer(run_dirs: list[Path], historical_root: Path | None = None,
                             overview_output_root: Path | None = None) -> None:
    st.subheader('Results at a glance')
    mode = st.radio("Evaluation", ['Automatic summary', 'Historical overview', 'Expert view'],
                    horizontal=True, key="result_view")
    if mode == 'Expert view':
        _render_expert(run_dirs, historical_root, overview_output_root)
        return
    root = historical_root or Path("data_results_csv")
    files = sorted(root.glob("*.csv"))
    signatures = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    archive, audit = _load_archive(str(root), signatures, original_rows=mode == 'Historical overview')
    if mode == 'Historical overview':
        if archive.empty:
            st.info('No historical classification results found.')
            return
        _historical_overview(archive, MatrixSettings(normalize=False),
                             overview_output_root or Path.cwd() / "artifacts/historical_overview", use_saved=False)
        return
    paths = [p for p in run_dirs if run_scope(p) != "modelnet40_legacy_full" and (p / "results.csv").is_file()]
    run_signatures = tuple(((p / "results.csv").stat().st_mtime_ns, (p / "results.csv").stat().st_size) for p in paths)
    merged = _load_runs(tuple(map(str, paths)), run_signatures)
    if not merged.empty:
        merged = merged[~merged["provider"].astype(str).str.casefold().str.contains("mock", regex=False)]
    merged = pd.concat([merged, archive], ignore_index=True, sort=False)
    if merged.empty:
        st.info('Key findings appear here automatically once results are available.')
        return
    datasets = sorted(merged["dataset"].unique(), key=lambda value: ({"modelnet": 0, "s3dis": 1}.get(value, 2), value))
    dataset = st.selectbox('Summary dataset', datasets, key="digest_dataset")
    data = prepare_confusion_rows(merged, dataset, paired=False)
    scopes = sorted(data["protocol_scope"].unique(), key=lambda value: (value != "current_paper_protocol", value))
    labels = {"current_paper_protocol": 'Current experiments', "modelnet33_exact_ids": 'Historical comparison runs',
              "historical_modelnet_pilot": "Historical pilot test", "historical_modelnet_full": "Historical full test",
              "historical_blk2go_pilot": "Historical BLK2GO test"}
    scope = st.selectbox("Experiment series", scopes, format_func=lambda value: labels.get(value, value), key="digest_scope")
    digest = automatic_digest(data[data["protocol_scope"].eq(scope)])
    if not digest or digest.get("empty_cohort"):
        st.info('There is no shared object cohort for an automatic comparison. Individual runs are available in the expert view.')
        return
    ranking, summary = digest["ranking"], digest["summary"]
    best = ranking.iloc[0]
    condition_names = {"views_1": '1 view', "views_2": '2 views', "views_3": '3 views', "views_4": '4 views',
                       "rgb": "RGB", "rgb_aabb": "RGB + AABB"}
    final_name = condition_names.get(digest["condition"], digest["condition"])
    one, two, three = st.columns(3)
    one.metric("Best accuracy · " + final_name, f"{best['accuracy']:.1%}")
    two.metric('Shared objects', f"{int(best['objects']):,}")
    three.metric('Models compared', len(ranking))
    st.write(f"**Best result with {final_name}: {best['model_label']}** with {best['accuracy']:.1%} correct responses and {best['coverage']:.1%} valid responses.")
    summary = summary.assign(Condition=summary["condition"].map(condition_names).fillna(summary["condition"]))
    chart = px.bar(summary, x='Condition', y="accuracy", color="model_label", barmode="group",
                   category_orders={'Condition': list(condition_names.values())},
                   labels={"accuracy": "Accuracy", "model_label": 'Model'},
                   hover_data=["objects", "observations", "coverage"], title='Model comparison across conditions')
    chart.update_layout(template="simple_white", legend_title_text="")
    chart.update_yaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(chart, width="stretch")
    gains = digest["gains"]
    if not gains.empty:
        strongest = gains.iloc[0]
        st.write(f"**Largest net improvement: {strongest['model_label']}**, {strongest['delta'] * 100:+.1f} percentage points from "
                 f"{condition_names.get(strongest['before'], strongest['before'])} to {condition_names.get(strongest['after'], strongest['after'])}. "
                 f"{int(strongest['gains']):,} responses became correct, {int(strongest['losses']):,} previously correct responses became incorrect "
                 f"({int(strongest['paired_observations']):,} object/repetition pairs).")
        chart = px.bar(gains, x="delta", y="model_label", orientation="h", hover_data=["before", "after", "gains", "losses", "paired_observations"],
                       labels={"delta": 'Accuracy change', "model_label": ""}, title='What changes with additional views or AABB dimensions?')
        chart.update_layout(template="simple_white"); chart.update_xaxes(tickformat="+.0%")
        st.plotly_chart(chart, width="stretch")
    difficult, stable = digest["difficult"], digest["stable"]
    if difficult.empty:
        st.write('**Consistently difficult classes:** No class with at least 5 objects per panel has a recognition rate below 50% in every compared panel.')
    else:
        st.write('**Consistently difficult:** ' + ", ".join(difficult["label"]) + '. Recognition rate below 50% in every compared model/condition panel, with at least 5 objects per class and panel.')
        chart = px.bar(difficult, x="mean_recall", y="label", orientation="h",
                       hover_data=["min_support", "min_recall", "max_recall"],
                       labels={"mean_recall": 'Mean recognition rate', "label": ""}, title='Classes with consistently low recognition')
        chart.update_layout(template="simple_white"); chart.update_xaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(chart, width="stretch")
    if len(stable):
        st.caption(f"\\* Filtered as stable: {len(stable)} consistently well-recognised classes. Complete values are available under source data.")
    pairs = digest["pairs"]
    if not pairs.empty:
        chart = px.bar(pairs.assign(Confusion=pairs["true_label"] + " → " + pairs["prediction"]),
                       x="errors", y="Confusion", orientation="h", hover_data=["support", "share"],
                       labels={"errors": 'Incorrect responses'}, title=f"Most frequent confusions · {best['model_label']} · {final_name}")
        chart.update_layout(template="simple_white", yaxis={"autorange": "reversed"})
        st.plotly_chart(chart, width="stretch")
    else:
        st.write('No confusions among classes with at least 5 observations in the detailed condition.')
    with st.expander('Show confusion matrices'):
        settings = MatrixSettings(focus="manual", order="alphabetical")
        shown = [label for label in digest["profiles"]["label"] if label not in set(stable["label"])]
        reference = make_reference(digest["data"], settings, shown)
        reference["other_name"] = f"* {len(stable)} stable classes"
        st.plotly_chart(confusion_figure(digest["data"], reference, settings, title="Confusion matrices"), width="stretch")
        if len(stable):
            st.caption('\\* Consistently well-recognised classes are combined. Difficult and rare classes remain visible. The combined diagonal is not class accuracy.')
    with st.expander('Source data and complete values'):
        st.caption(f"The comparison uses the same protocol and object IDs shared across all available panels. Series below 95% of the largest minimum panel size are hidden: {digest['excluded_series']}. "
                   f"Total {digest['excluded_observations']:,} rows excluded by series/object selection. Missing conditions are not filled in; errors count as incorrect. Repetitions are not independent objects. Differences are descriptive, with no significance claim.")
        st.dataframe(summary, hide_index=True, width="stretch")
        st.caption('Consistently well-recognised: recognition rate at least 80%, range at most 5 percentage points, and at least 95% unchanged object predictions across two or more panels, with at least 5 objects per class and panel.')
        st.caption('The overview shows up to five difficult classes and five confusions. * marks filtered, consistently well-recognised classes only. Ties for the best model are resolved alphabetically.')
        st.write(", ".join(stable["label"]) or 'No class meets every stability criterion.')
        st.dataframe(digest["profiles"], hide_index=True, width="stretch")
        if not gains.empty:
            st.dataframe(gains, hide_index=True, width="stretch")
        if not audit.empty:
            st.dataframe(audit, hide_index=True, width="stretch")
    st.download_button('Download summary as CSV', summary.to_csv(index=False).encode("utf-8"),
                       file_name=f"{dataset}_summary.csv", mime="text/csv", key="digest_download")
