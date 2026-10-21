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

FOCUS = {"all": "Alle Klassen", "errors": "Schwierigste Klassen (Top N)",
         "changed": "Größte Recall-Unterschiede (Top N)", "compact": "Stabile Klassen zusammenfassen",
         "stable_good": "Stabil gut erkannte Klassen", "stable_bad": "Stabil schwierige Klassen",
         "manual": "Klassen selbst auswählen"}
ORDER = {"alphabetical": "Alphabetisch", "recall": "Schwierige Klassen zuerst", "cluster": "Nach Verwechslungen clustern"}
GROUPING = {"none": "Einzelklassen", "semantic": "Semantische Vorschläge", "cluster": "Verwechslungscluster", "custom": "Eigene Gruppen"}


@st.cache_data(show_spinner=False)
def _load_runs(paths: tuple[str, ...], signatures: tuple[tuple[int, int], ...]) -> pd.DataFrame:
    del signatures
    return merge_run_results(paths)


@st.cache_data(show_spinner=False)
def _profiles(data: pd.DataFrame) -> pd.DataFrame:
    return class_profiles(data)


def _validate_preset(preset: dict) -> MatrixSettings:
    if preset.get("schema_version") != 1 or not isinstance(preset.get("dataset"), str):
        raise ValueError("Unbekanntes Referenzformat.")
    order, shown, groups = preset["order"], preset["shown"], preset["groups"]
    if not isinstance(order, list) or not 0 < len(order) <= 200 or not all(isinstance(label, str) for label in order):
        raise ValueError("Ungültige Klassenreihenfolge.")
    if len(set(order)) != len(order) or not isinstance(shown, list) or not set(shown).issubset(order):
        raise ValueError("Ungültige Fokusklassen.")
    if not isinstance(groups, dict) or set(groups) != set(order) or not all(isinstance(value, str) and value.strip() for value in groups.values()):
        raise ValueError("Ungültige Gruppen.")
    settings = MatrixSettings(**preset["settings"])
    if settings.focus not in FOCUS or settings.order not in ORDER or settings.grouping not in GROUPING:
        raise ValueError("Unbekannte Darstellungsoption.")
    for name in ("stable_fraction", "recall_tolerance", "good_recall", "bad_recall"):
        if not isinstance(getattr(settings, name), (int, float)) or not 0 <= getattr(settings, name) <= 1:
            raise ValueError("Schwellenwerte müssen zwischen 0 und 1 liegen.")
    if not isinstance(settings.mask_below, (float, int)) or not 0 <= settings.mask_below <= (1 if settings.normalize else 100000):
        raise ValueError("Ungültige Zellmaske.")
    if not 1 <= settings.top_k <= 100 or not 1 <= settings.min_support <= 100000 or not 1 <= settings.clusters <= 40:
        raise ValueError("Unzulässige Anzahl Klassen, Gruppen oder Beobachtungen.")
    for name in ("normalize", "errors_only", "include_invalid"):
        if not isinstance(getattr(settings, name), bool):
            raise ValueError("Ungültige Anzeigeoption.")
    return settings


@st.cache_data(show_spinner=False)
def _load_archive(path: str, signatures: tuple[tuple[str, int, int], ...], original_rows: bool = False):
    del signatures
    return load_historical_csvs(path, original_rows=original_rows)


def _historical_overview(archive: pd.DataFrame, settings: MatrixSettings, output_root: Path, *, use_saved: bool = True) -> None:
    st.subheader("Historische Gesamtübersicht wie in der Vorlage")
    real_model = st.selectbox("BLK2GO-Modell in der unteren Reihe", ["gemini-1.5-pro", "gpt-4o"], key="history_real_model")
    one, two = st.columns(2)
    show_values = one.checkbox("Zellwerte in der historischen Tafel", value=True, key="history_values")
    show_zeros = two.checkbox("Auch Nullwerte beschriften", value=False, key="history_zeros")
    saved = st.session_state.get("confusion_display_settings", {}) if use_saved else {}
    references = {dataset: (item[0], item[1]) for dataset, item in saved.items() if dataset in {"modelnet", "blk2go"}}
    panels, references = overview_panels(archive, settings, references, real_model)
    figure = historical_overview_figure(panels, references, show_values=show_values, show_zeros=show_zeros)
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False,
        "toImageButtonOptions": {"format": "svg", "filename": "historical_overview", "scale": 3}})
    table = pd.DataFrame([{**{key: panel.get(key) for key in ["row", "col", "dataset", "model", "scope", "condition", "objects", "observations"]},
                           "state": "loaded" if "matrix" in panel else "not reconstructed"} for panel in panels])
    with st.expander("Historische Panels und Datenbasis"):
        st.caption("ModelNet wie im Originalskript: jede CSV-Zeile zählt, Klassenname vor dem ersten Unterstrich. Mehrere Ansichten sind keine zusätzlichen unabhängigen Objekte. Zusammenfassungen verwenden dagegen Objektentscheidungen. Pilot, Full-Test und BLK2GO bleiben getrennt; gemeinsame Skalen je Datensatz. Gruppendiagonalen sind keine Klassen-Accuracy. Stable Diffusion bleibt ohne eindeutige Klassenantworten ein Platzhalter.")
        st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Historische Tafel als interaktives HTML", figure.to_html(include_plotlyjs=True).encode("utf-8"),
                        file_name="historical_overview.html", mime="text/html", key="history_html")
    st.download_button("Historische Matrixwerte als CSV", overview_values(panels).to_csv(index=False).encode("utf-8"),
                        file_name="historical_overview.csv", mime="text/csv", key="history_csv")
    if st.button("Historisches Grafikpaket mit SVG und PNG erzeugen", key="history_export"):
        with st.spinner("Historische Tafel als Vektorgrafik und hochauflösendes Bild erzeugen …"):
            try:
                paths = save_historical_overview(output_root, panels, references, figure,
                                                  show_values=show_values, show_zeros=show_zeros)
                output = io.BytesIO()
                with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                    for path in paths.values():
                        bundle.write(path, arcname=path.name)
                st.session_state["history_bundle"] = output.getvalue()
                st.session_state["history_bundle_source"] = str(output_root)
                st.success(f"SVG, PNG, HTML, CSV und Referenz-JSON gespeichert: {output_root}")
            except (ImportError, OSError, ValueError) as exc:
                st.error(f"Grafikpaket konnte nicht erzeugt werden: {exc}")
    if "history_bundle" in st.session_state:
        st.download_button("Zuletzt erzeugtes historisches Grafikpaket herunterladen", st.session_state["history_bundle"],
                           file_name="historical_overview.zip", mime="application/zip", key="history_zip")
        st.caption("Das ZIP enthält den zuletzt ausdrücklich erzeugten Stand. Nach Filteränderungen das Grafikpaket neu erzeugen.")


def _render_expert(run_dirs: list[Path], historical_root: Path | None = None,
                             overview_output_root: Path | None = None) -> None:
    st.subheader("Einheitlicher Confusion-Matrix-Explorer")
    st.caption("Dieselben Filter, Klassenreihenfolgen und Farbskalen für alte und neue Ergebnisse. ModelNet und S3DIS werden getrennt ausgewertet.")
    if not st.checkbox("Matrix-Explorer aktivieren", value=False, key="cm_enabled"):
        return
    default = [path for path in run_dirs if run_scope(path) != "modelnet40_legacy_full"]
    selected = st.multiselect("Matrix-Explorer: Runs", run_dirs, default=default, format_func=lambda path: path.name,
                             key="cm_runs", help="Auch importierte Altresultate können gewählt werden. Der vollständige ModelNet40-Audit ist bewusst separat auswählbar.")
    signatures = tuple(((path / "results.csv").stat().st_mtime_ns, (path / "results.csv").stat().st_size) for path in selected)
    with st.spinner("Ergebnisse für gemeinsame Matrixfilter laden …"):
        merged = _load_runs(tuple(str(path) for path in selected), signatures)
    archive_root = Path(st.text_input("Historischer Paper-CSV-Ordner", str(historical_root or Path("data_results_csv")), key="cm_archive_path"))
    archive_enabled = st.checkbox("Historische Paper-CSV-Dateien einbeziehen", value=archive_root.is_dir(), key="cm_archive_enabled")
    archive = pd.DataFrame()
    if archive_enabled:
        archive_files = sorted(archive_root.glob("*.csv"))
        archive, audit = _load_archive(str(archive_root), tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size) for path in archive_files))
        with st.expander("Historische Quelldateien und Rekonstruierbarkeit"):
            st.dataframe(audit, hide_index=True, width="stretch")
        if not archive.empty:
            merged = pd.concat([merged, archive], ignore_index=True, sort=False)
    historical_layout = False
    if not archive.empty:
        historical_layout = st.radio("Matrixdarstellung", ["Gewählte Modelle und Bedingungen", "Historische Gesamtübersicht (wie Vorlage)"],
                                     key="cm_layout", horizontal=True) == "Historische Gesamtübersicht (wie Vorlage)"
        if historical_layout:
            archive, audit = _load_archive(str(archive_root), tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size)
                                         for path in sorted(archive_root.glob("*.csv"))), original_rows=True)
            merged = archive.copy()
            _historical_overview(archive, MatrixSettings(normalize=False),
                                 overview_output_root or Path.cwd() / "artifacts/historical_overview", use_saved=False)
            return
    if merged.empty:
        st.info("Diese Runs enthalten keine Ergebniszeilen.")
        return
    dataset = st.selectbox("Matrix-Explorer: Datensatz", sorted(merged["dataset"].astype(str).unique()), key="cm_dataset")
    candidates = prepare_confusion_rows(merged, dataset, paired=False)
    options = sorted(candidates["series_id"].unique())
    panel_sizes = candidates.groupby(["series_id", "panel_id"])["sample_id"].nunique()
    minimum_sizes = panel_sizes.groupby("series_id").min()
    default_series = options if historical_layout else minimum_sizes[minimum_sizes.ge(minimum_sizes.max() * .95)].index.tolist()
    reference_series = st.multiselect("Modelle und Protokolle der Referenzbasis", options, default=default_series,
                                     key="cm_reference_series", help="Vollständige Reihen sind voreingestellt. Pilot- und Teilläufe können ausdrücklich hinzugefügt werden; ihre gemeinsamen Objekt-IDs können den Vergleich verkleinern.")
    candidates = candidates[candidates["series_id"].isin(reference_series)]
    if candidates.empty:
        st.info("Mindestens eine Modell-/Protokollreihe für die Referenz wählen.")
        return
    paired = (False if historical_layout else st.checkbox("Nur gemeinsame Objekt-IDs der gewählten Panels", value=True, key="cm_paired",
                         help="Schließt API-Fehler nicht aus. Verwendet dieselben Objekte für den Vergleich der vorhandenen Modell-/Protokoll-/Bedingungspanels."))
    data = prepare_confusion_rows(candidates, dataset, paired=paired)
    if data.empty:
        st.warning("Keine gemeinsamen Objekt-IDs. Die Run-Auswahl anpassen oder den gemeinsamen Objektfilter deaktivieren.")
        return
    st.caption(f"Referenzbasis: {data['sample_id'].nunique():,} Objekt-IDs · {len(data):,} Beobachtungen · {data['panel_id'].nunique()} Panels. Fehlende Bedingungen werden nicht als Nullergebnis dargestellt.")
    with st.expander("Gemeinsame Filter, Cluster und Referenz", expanded=False):
        uploaded = st.file_uploader("Gespeicherte Matrix-Referenz laden", type=["json"], key="cm_upload")
        if uploaded is not None and st.button("Matrix-Referenz anwenden", key="cm_apply"):
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
                    raise ValueError("Die gespeicherte Referenz gehört zu einem anderen Datensatz.")
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
                st.success("Referenz geladen. Reihenfolge und Fokus bleiben für die ausgewählten Modelle gleich.")
            except (ValueError, TypeError, KeyError) as exc:
                st.error(f"Referenz konnte nicht geladen werden: {exc}")
        first, second, third = st.columns(3)
        focus = first.selectbox("Klassenfokus", list(FOCUS), format_func=FOCUS.get, key="cm_focus")
        top_k = int(second.number_input("Top N Klassen", min_value=1, max_value=100, value=12, key="cm_top_k"))
        min_support = int(third.number_input("Mindestzahl je Klasse und Panel", min_value=1, max_value=100000, value=5, key="cm_min_support"))
        first, second = st.columns(2)
        order = first.selectbox("Gemeinsame Klassenreihenfolge", list(ORDER), format_func=ORDER.get, key="cm_order")
        grouping = second.selectbox("Klassen gruppieren", list(GROUPING), format_func=GROUPING.get, key="cm_grouping")
        clusters = int(st.number_input("Anzahl Verwechslungscluster", min_value=1, max_value=40, value=6, key="cm_clusters"))
        labels = sorted(set(data["true_label"]) | (set(data["prediction"]) - {"[No valid response]"}))
        reference_saved = st.session_state.get(f"cm_reference_{dataset}", {})
        options = sorted(set(labels) | set(reference_saved.get("order", [])))
        manual = st.multiselect("Eigene Fokusklassen", options, default=options, key="cm_manual") if focus == "manual" else []
        custom = None
        if grouping in {"custom", "semantic"}:
            seed = st.session_state.get("cm_custom_seed", semantic_groups(options, dataset))
            edited = st.data_editor(pd.DataFrame([{"Klasse": label, "Gruppe": seed.get(label, label)} for label in options]),
                                    disabled=["Klasse"], hide_index=True, key="cm_group_editor")
            custom = dict(zip(edited["Klasse"], edited["Gruppe"].fillna("").astype(str)))
            if grouping == "semantic":
                # Semantic suggestions are editable; use the edited mapping for the actual result.
                st.caption("Vorschläge für die Exploration, keine fachlich validierte neue Taxonomie. Gruppen können hier umbenannt werden.")
        first, second = st.columns(2)
        stable_fraction = first.slider("Unveränderte Objektvorhersagen mindestens", 0., 1., .95, .01, key="cm_stable_fraction")
        recall_tolerance = second.slider("Maximale Recall-Spanne für stabile Klassen", 0., 1., .05, .01, key="cm_recall_tolerance")
        first, second = st.columns(2)
        good_recall = first.slider("Stabil gut: Recall mindestens", 0., 1., .8, .05, key="cm_good_recall")
        bad_recall = second.slider("Stabil schwierig: Recall höchstens", 0., 1., .2, .05, key="cm_bad_recall")
        first, second, third = st.columns(3)
        normalize = first.checkbox("Zeilenanteile statt Anzahlen", value=True, key="cm_normalize")
        errors_only = second.checkbox("Nur Verwechslungen zeigen", value=False, key="cm_errors_only")
        include_invalid = third.checkbox("Fehlende/ungültige Antworten zeigen", value=True, key="cm_include_invalid")
        mask_below = (st.slider("Kleine Zellen unter diesem Anteil ausblenden", 0., 1., 0., .01, key="cm_mask_below",
                               help="Nur Anzeigemaske; Werte, Nenner und exportierte Anzahlen bleiben unverändert.") if normalize else
                      float(st.number_input("Kleine Zellen unter dieser Anzahl ausblenden", min_value=0, max_value=100000, value=0, key="cm_count_mask")))
        lock = st.checkbox("Referenz bei Run- und Modellwechsel festhalten", value=True, key="cm_lock")
        refresh = st.button("Fokus und Cluster aus dieser Referenzbasis neu berechnen", key="cm_refresh")
    settings = MatrixSettings(focus, top_k, min_support, stable_fraction, recall_tolerance, good_recall,
                              bad_recall, order, grouping, clusters, normalize, errors_only, mask_below, include_invalid)
    signature = json.dumps({"settings": settings.public(), "manual": manual, "custom": custom}, sort_keys=True)
    reference_key, signature_key = f"cm_reference_{dataset}", f"cm_signature_{dataset}"
    saved_signature = st.session_state.get(signature_key)
    if (reference_key not in st.session_state or not lock or refresh
            or (saved_signature is not None and saved_signature != signature)):
        with st.spinner("Klassenprofile und gemeinsame Cluster berechnen …"):
            reference = make_reference(data, settings, manual, custom)
        st.session_state[reference_key] = reference
    reference = st.session_state[reference_key]
    st.session_state[signature_key] = signature
    missing_labels = set(labels) - set(reference["order"])
    if missing_labels:
        st.warning("Neue Klassen fehlen in der festgehaltenen Referenz: " + ", ".join(sorted(missing_labels)) + ". Bitte die Referenz neu berechnen.")
        return
    st.session_state.setdefault("confusion_display_settings", {})[dataset] = (settings, reference, data)
    st.caption("Fokus und Reihenfolge werden aus der gemeinsamen Referenz bestimmt, nicht für jedes angezeigte Modell neu. Explizite Filteränderungen berechnen den Fokus neu; der Modellwähler unten ändert ihn nicht.")
    if historical_layout:
        _historical_overview(archive, settings, overview_output_root or Path.cwd() / "artifacts" / "historical_overview")
        return
    series_options = sorted(data["series_id"].unique())
    shown_series = st.multiselect("Angezeigte Modelle und Protokolle", series_options,
                                  default=series_options[:2], key="cm_series")
    displayed = data[data["series_id"].isin(shown_series)]
    if displayed.empty:
        st.info("Mindestens ein Modell/Protokoll zum Anzeigen auswählen.")
        return
    totals = st.columns(3)
    totals[0].metric("Accuracy – vollständige Auswahl", f"{displayed['is_correct'].mean():.1%}")
    totals[1].metric("Coverage – vollständige Auswahl", f"{displayed['prediction'].ne('[No valid response]').mean():.1%}")
    totals[2].metric("Explizit gezeigte Klassen", f"{len(reference['shown'])}/{len(reference['order'])}")
    st.caption("Die Kennzahlen oben nutzen alle Beobachtungen der angezeigten Panels vor dem Klassenfokus. Bei mehreren Modellen sind sie beobachtungsgewichtet; Einzelwerte stehen in der bestehenden Run-Auswertung.")
    if settings.grouping != "none" or len(reference["shown"]) != len(reference["order"]):
        st.info("Gruppen und „Other“ sammeln Originalklassen. Ihre Diagonale enthält auch Verwechslungen innerhalb einer Gruppe und ist keine Klassen-Accuracy. Im Hover und CSV bleiben ursprünglich richtige und falsche Antworten getrennt ausgewiesen.")
    if not settings.include_invalid:
        st.caption("Die Matrix zeigt nur gültige Antworten; ihre Zeilenanteile sind bedingt auf gültige Antworten. Accuracy und Coverage oben bleiben auf der vollständigen Auswahl.")
    figure = confusion_figure(displayed, reference, settings, title=f"{dataset.upper()} · shared class reference")
    export_format = st.selectbox("Grafik-Downloadformat", ["svg", "png"], key="cm_export_format",
                                 help="Der Kamera-Button in der Grafik exportiert SVG als Vektorgrafik oder PNG in hoher Auflösung.")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "toImageButtonOptions": {
        "format": export_format, "filename": f"{dataset}_confusion_filtered", "scale": 3}})
    st.caption("Graue Zellen: keine Beobachtungen oder durch die Anzeigemaske ausgeblendet. Weiß entspricht einem tatsächlich gemessenen Nullwert. Die SVG-/PNG-Ausgabe erfolgt über den Kamera-Button der Grafik.")
    with st.expander("Klassenprofile, stabile Platzhalter und Gruppenzuordnung"):
        profiles = _profiles(data)
        profiles["shown"] = profiles["label"].isin(reference["shown"])
        profiles["group"] = profiles["label"].map(reference["groups"])
        st.dataframe(profiles, hide_index=True, width="stretch")
        st.caption("Stabilität setzt mindestens zwei Panels, genügend Beispiele in jedem Panel, eine kleine Recall-Spanne und unveränderte Vorhersagen derselben Objekt-IDs voraus. Seltene Klassen werden nicht automatisch als stabil versteckt.")
    changes = paired_changes(displayed)
    if not changes.empty:
        with st.expander("Gewinne und Verluste auf denselben Objekten", expanded=False):
            st.caption("ModelNet: erste gegen letzte vorhandene Ansichtszahl; S3DIS: RGB gegen RGB+AABB. Nur eindeutig zuordenbare Objekt- und Wiederholungspaare; keine Signifikanzbehauptung.")
            chart = px.bar(changes, x="label", y="delta_recall", color="series_id", barmode="group",
                           hover_data=["paired_observations", "gains", "losses", "before", "after"],
                           labels={"label": "Class", "delta_recall": "Recall change", "series_id": "Model / protocol"})
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
    one.download_button("Matrix als interaktives HTML", figure.to_html(include_plotlyjs=True).encode("utf-8"),
                        file_name=f"{dataset}_confusion.html", mime="text/html", key="cm_html")
    two.download_button("Matrixwerte und Originalfehler als CSV", exported.to_csv(index=False).encode("utf-8"),
                        file_name=f"{dataset}_confusion.csv", mime="text/csv", key="cm_csv")
    three.download_button("Filter und Referenz speichern", json.dumps(preset, ensure_ascii=False, indent=2).encode("utf-8"),
                          file_name=f"{dataset}_confusion_reference.json", mime="application/json", key="cm_json")


def render_confusion_explorer(run_dirs: list[Path], historical_root: Path | None = None,
                             overview_output_root: Path | None = None) -> None:
    st.subheader("Ergebnisse auf einen Blick")
    mode = st.radio("Auswertung", ["Automatische Zusammenfassung", "Historische Gesamtgrafik", "Expertenansicht"],
                    horizontal=True, key="result_view")
    if mode == "Expertenansicht":
        _render_expert(run_dirs, historical_root, overview_output_root)
        return
    root = historical_root or Path("data_results_csv")
    files = sorted(root.glob("*.csv"))
    signatures = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    archive, audit = _load_archive(str(root), signatures, original_rows=mode == "Historische Gesamtgrafik")
    if mode == "Historische Gesamtgrafik":
        if archive.empty:
            st.info("Keine historischen Klassifikationsergebnisse gefunden.")
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
        st.info("Sobald Ergebnisse vorliegen, erscheinen hier automatisch die wichtigsten Befunde.")
        return
    datasets = sorted(merged["dataset"].unique(), key=lambda value: ({"modelnet": 0, "s3dis": 1}.get(value, 2), value))
    dataset = st.selectbox("Datensatz der Zusammenfassung", datasets, key="digest_dataset")
    data = prepare_confusion_rows(merged, dataset, paired=False)
    scopes = sorted(data["protocol_scope"].unique(), key=lambda value: (value != "current_paper_protocol", value))
    labels = {"current_paper_protocol": "Aktuelle Experimente", "modelnet33_exact_ids": "Frühere Vergleichsläufe",
              "historical_modelnet_pilot": "Historischer Pilottest", "historical_modelnet_full": "Historischer Full-Test",
              "historical_blk2go_pilot": "Historischer BLK2GO-Test"}
    scope = st.selectbox("Versuchsreihe", scopes, format_func=lambda value: labels.get(value, value), key="digest_scope")
    digest = automatic_digest(data[data["protocol_scope"].eq(scope)])
    if not digest or digest.get("empty_cohort"):
        st.info("Keine gemeinsame Objektbasis für eine automatische Gegenüberstellung. Einzelne Läufe sind in der Expertenansicht verfügbar.")
        return
    ranking, summary = digest["ranking"], digest["summary"]
    best = ranking.iloc[0]
    condition_names = {"views_1": "1 Ansicht", "views_2": "2 Ansichten", "views_3": "3 Ansichten", "views_4": "4 Ansichten",
                       "rgb": "RGB", "rgb_aabb": "RGB + AABB"}
    final_name = condition_names.get(digest["condition"], digest["condition"])
    one, two, three = st.columns(3)
    one.metric("Beste Accuracy · " + final_name, f"{best['accuracy']:.1%}")
    two.metric("Gemeinsame Objekte", f"{int(best['objects']):,}")
    three.metric("Verglichene Modelle", len(ranking))
    st.write(f"**Bestes Ergebnis bei {final_name}: {best['model_label']}** mit {best['accuracy']:.1%} richtig klassifizierten Antworten und {best['coverage']:.1%} gültigen Antworten.")
    summary = summary.assign(Bedingung=summary["condition"].map(condition_names).fillna(summary["condition"]))
    chart = px.bar(summary, x="Bedingung", y="accuracy", color="model_label", barmode="group",
                   category_orders={"Bedingung": list(condition_names.values())},
                   labels={"accuracy": "Accuracy", "model_label": "Modell"},
                   hover_data=["objects", "observations", "coverage"], title="Modellvergleich über die Bedingungen")
    chart.update_layout(template="simple_white", legend_title_text="")
    chart.update_yaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(chart, width="stretch")
    gains = digest["gains"]
    if not gains.empty:
        strongest = gains.iloc[0]
        st.write(f"**Beste Veränderungsbilanz: {strongest['model_label']}**, {strongest['delta'] * 100:+.1f} Prozentpunkte von "
                 f"{condition_names.get(strongest['before'], strongest['before'])} zu {condition_names.get(strongest['after'], strongest['after'])}. "
                 f"{int(strongest['gains']):,} Antworten wurden richtig, {int(strongest['losses']):,} zuvor richtige wurden falsch "
                 f"({int(strongest['paired_observations']):,} Objekt-/Wiederholungspaare).")
        chart = px.bar(gains, x="delta", y="model_label", orientation="h", hover_data=["before", "after", "gains", "losses", "paired_observations"],
                       labels={"delta": "Accuracy-Veränderung", "model_label": ""}, title="Was bringen zusätzliche Ansichten oder AABB?")
        chart.update_layout(template="simple_white"); chart.update_xaxes(tickformat="+.0%")
        st.plotly_chart(chart, width="stretch")
    difficult, stable = digest["difficult"], digest["stable"]
    if difficult.empty:
        st.write("**Dauerhaft schwierige Klassen:** Keine Klasse mit mindestens 5 Objekten je Panel liegt in allen verglichenen Panels unter 50 % Recall.")
    else:
        st.write("**Dauerhaft schwierig:** " + ", ".join(difficult["label"]) + ". In jedem verglichenen Modell-/Bedingungspanel unter 50 % Recall (mindestens 5 Objekte je Klasse und Panel).")
        chart = px.bar(difficult, x="mean_recall", y="label", orientation="h",
                       hover_data=["min_support", "min_recall", "max_recall"],
                       labels={"mean_recall": "Mittlerer Recall", "label": ""}, title="Klassen mit durchgehend niedriger Erkennung")
        chart.update_layout(template="simple_white"); chart.update_xaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(chart, width="stretch")
    if len(stable):
        st.caption(f"\\* Ausgefiltert, da unauffällig: {len(stable)} stabil gut erkannte Klassen. Vollständige Werte unter Datenbasis.")
    pairs = digest["pairs"]
    if not pairs.empty:
        chart = px.bar(pairs.assign(Verwechslung=pairs["true_label"] + " → " + pairs["prediction"]),
                       x="errors", y="Verwechslung", orientation="h", hover_data=["support", "share"],
                       labels={"errors": "Falsche Antworten"}, title=f"Häufigste Verwechslungen · {best['model_label']} · {final_name}")
        chart.update_layout(template="simple_white", yaxis={"autorange": "reversed"})
        st.plotly_chart(chart, width="stretch")
    else:
        st.write("Keine Verwechslungen bei Klassen mit mindestens 5 Beobachtungen in der Detailbedingung.")
    with st.expander("Konfusionsmatrizen anzeigen"):
        settings = MatrixSettings(focus="manual", order="alphabetical")
        shown = [label for label in digest["profiles"]["label"] if label not in set(stable["label"])]
        reference = make_reference(digest["data"], settings, shown)
        reference["other_name"] = f"* {len(stable)} stable classes"
        st.plotly_chart(confusion_figure(digest["data"], reference, settings, title="Konfusionsmatrizen"), width="stretch")
        if len(stable):
            st.caption("\\* Stabil gut erkannte Klassen zusammengefasst. Schwierige und seltene Klassen bleiben sichtbar. Die Sammeldiagonale ist keine Klassen-Accuracy.")
    with st.expander("Datenbasis und vollständige Zahlen"):
        st.caption(f"Gleiches Protokoll und gemeinsame Objekt-IDs aller vorhandenen Panels. Serien unter 95 % der größten Mindest-Panelgröße sind ausgeblendet: {digest['excluded_series']}. "
                   f"Insgesamt {digest['excluded_observations']:,} Zeilen durch Serien-/Objektauswahl ausgeschlossen. Fehlende Bedingungen werden nicht ergänzt; Fehler zählen als falsch. Wiederholungen sind keine unabhängigen Objekte. Unterschiede sind beschreibend, ohne Signifikanzbehauptung.")
        st.dataframe(summary, hide_index=True, width="stretch")
        st.caption("Stabil gut: mindestens 80 % Recall, höchstens 5 Prozentpunkte Spanne und mindestens 95 % unveränderte Objektvorhersagen über mindestens zwei Panels; mindestens 5 Objekte je Klasse und Panel.")
        st.caption("Die Übersicht zeigt höchstens fünf schwierige Klassen und fünf Verwechslungen. * kennzeichnet ausschließlich ausgefilterte, stabil gut erkannte Klassen. Bei gleichem Spitzenwert wird ein Modell alphabetisch ausgewählt.")
        st.write(", ".join(stable["label"]) or "Keine Klasse erfüllt alle Stabilitätskriterien.")
        st.dataframe(digest["profiles"], hide_index=True, width="stretch")
        if not gains.empty:
            st.dataframe(gains, hide_index=True, width="stretch")
        if not audit.empty:
            st.dataframe(audit, hide_index=True, width="stretch")
    st.download_button("Zusammenfassung als CSV", summary.to_csv(index=False).encode("utf-8"),
                       file_name=f"{dataset}_summary.csv", mime="text/csv", key="digest_download")
