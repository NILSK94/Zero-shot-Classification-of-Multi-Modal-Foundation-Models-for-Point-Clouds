from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from mmfm_tool.constants import MODELNET_CLASSES, S3DIS_CLASSES
from mmfm_tool.datasets import (
    image_paths_from_row,
    load_prepared_s3dis_manifest,
    scan_modelnet_manifest,
    scan_s3dis_annotations,
    select_paired_modelnet,
    select_s3dis_area,
)
from mmfm_tool.evaluation import load_results_csv
from mmfm_tool.figures import confusion_matrix_grid, model_only_legend
from mmfm_tool.confusion_ui import render_confusion_explorer
from mmfm_tool.legacy_results import LegacyResultError, import_legacy_results, load_legacy_results
from mmfm_tool.jev_checks import run_classification_preview
from mmfm_tool.local_paths import data_paths, find_paper_reference_figures
from mmfm_tool.providers import JevProvider, build_prompt, photo_check_scores
from mmfm_tool.paper_export import build_paper_export
from mmfm_tool.publication_ui import render_publication_export
from mmfm_tool.rendering import prepare_s3dis_views
from mmfm_tool.run_comparison import (
    PROTOCOL_LABELS,
    merge_run_results,
    modelnet_view_table,
    run_scope,
    summarize_merged_results,
)
from mmfm_tool.runner import (
    ExperimentSpec,
    ModelTarget,
    create_run_directory,
    planned_request_count,
    run_experiment,
)

PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_DATA_PATHS = data_paths(PROJECT_DIR)
DEFAULT_MODELNET = LOCAL_DATA_PATHS["modelnet"]
DEFAULT_S3DIS = LOCAL_DATA_PATHS["s3dis"]
DEFAULT_PREPARED = LOCAL_DATA_PATHS["prepared"]
DEFAULT_RUNS = LOCAL_DATA_PATHS["runs"]
DEFAULT_OLD_RESULTS = LOCAL_DATA_PATHS["legacy_results"]
DEFAULT_PAPER_EXPORT = LOCAL_DATA_PATHS["paper_export"]
DEFAULT_HISTORICAL_CSV = LOCAL_DATA_PATHS["historical_csv"]
DEFAULT_LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "")
DEFAULT_QWEN_MODEL = os.getenv("LITELLM_MODEL", "")

st.set_page_config(page_title="MMFM Lab", page_icon="◈", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem;}
      [data-testid="stMetricValue"] {font-size: 1.7rem;}
      .paper-note {border-left: 4px solid #4786ff; padding: .65rem 1rem; background: rgba(71,134,255,.08); border-radius: 0 .5rem .5rem 0;}
      .small-muted {color: #6b7280; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_modelnet(path: str) -> pd.DataFrame:
    return scan_modelnet_manifest(path)


@st.cache_data(show_spinner=False)
def cached_s3dis(path: str) -> pd.DataFrame:
    return scan_s3dis_annotations(path)


@st.cache_data(show_spinner=False)
def cached_prepared(path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return load_prepared_s3dis_manifest(path)


def split_models(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def configured_targets() -> list[ModelTarget]:
    targets: list[ModelTarget] = []
    if use_mock:
        targets.append(ModelTarget("mock", "mock-v1"))
    if use_gemini:
        targets.extend(ModelTarget("gemini", model, gemini_key) for model in split_models(gemini_models))
    if use_jev:
        targets.extend(ModelTarget("jev", model, openrouter_key) for model in split_models(jev_models))
    if use_litellm:
        targets.extend(
            ModelTarget(
                "litellm",
                model,
                litellm_key,
                litellm_base_url,
                max_concurrent_requests=int(litellm_max_concurrent_requests),
            )
            for model in split_models(litellm_models)
        )
    return targets


def prepared_frame(manifest_path: Path) -> pd.DataFrame:
    modified = manifest_path.stat().st_mtime_ns if manifest_path.exists() else 0
    return cached_prepared(str(manifest_path), modified)


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        # Preserve numeric NaN values. Replacing them with empty strings turns
        # metric columns into mixed object columns and breaks Arrow rendering.
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def show_photo_check(result: dict | pd.Series, threshold: float, key: str) -> None:
    if result.get("provider") != "jev_photo_check" or result.get("status") != "ok":
        return
    scores = photo_check_scores(str(result["raw_response"]))
    selected = str(result["pred_label"])
    table = pd.DataFrame([
        {"Klasse": label, "P(Ja)": score, "Vorhersage": label == selected,
         "Prüfschwelle erreicht": score >= threshold}
        for label, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ])
    st.caption("Jede Klasse wurde unabhängig geprüft. Die Ja-Wahrscheinlichkeiten ergeben zusammen nicht zwingend 100 %.")
    if scores[selected] < threshold:
        st.warning(f"Zum Prüfen markiert: {selected} erreicht {scores[selected]:.1%}, die Prüfschwelle liegt bei {threshold:.0%}. Die Vorhersage zählt weiterhin in der Auswertung.")
    elif sum(value >= threshold for value in scores.values()) > 1:
        st.warning("Mehrere Klassen erreichen die Prüfschwelle. Als Vorhersage zählt die Klasse mit dem höchsten P(Ja).")
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"P(Ja)": st.column_config.ProgressColumn("P(Ja)", min_value=0., max_value=1., format="%.3f")})
    st.download_button("Photo-Check-Werte herunterladen", table.to_csv(index=False).encode("utf-8"),
                       file_name="photo_check.csv", mime="text/csv", key=key)


@st.cache_data(show_spinner=False)
def cached_merged_run_results(
    run_paths: tuple[str, ...],
    result_signatures: tuple[tuple[int, int], ...],
    keep_duplicate_tasks: bool,
) -> pd.DataFrame:
    del result_signatures
    return merge_run_results(run_paths, keep_duplicate_tasks=keep_duplicate_tasks)


st.sidebar.header("Daten")
modelnet_path = Path(st.sidebar.text_input("ModelNet-Ordner", str(DEFAULT_MODELNET))).expanduser()
s3dis_path = Path(st.sidebar.text_input("S3DIS-Ordner", str(DEFAULT_S3DIS))).expanduser()
prepared_root = Path(st.sidebar.text_input("Vorbereitete S3DIS-Daten", str(DEFAULT_PREPARED))).expanduser()
runs_root = Path(st.sidebar.text_input("Ergebnisordner", str(DEFAULT_RUNS))).expanduser()
old_results_path = Path(st.sidebar.text_input("Alte Ergebnisdateien", str(DEFAULT_OLD_RESULTS))).expanduser()

st.sidebar.header("Modelle")
use_mock = st.sidebar.checkbox(
    "Kostenfreier Pipeline-Test",
    value=False,
    help="Erzeugt künstliche Testantworten ohne wissenschaftliche Aussage und ist deshalb standardmäßig deaktiviert.",
)
use_gemini = st.sidebar.checkbox("Google / Gemini", value=False)
gemini_models = st.sidebar.text_input("Gemini-Modell-ID(s)", os.getenv("GEMINI_MODEL", ""), help="Mehrere IDs durch Komma trennen.")
gemini_key = st.sidebar.text_input(
    "Gemini API-Key",
    value=os.getenv("GEMINI_API_KEY", ""),
    type="password",
    help="Wird nur im laufenden Prozess verwendet und nie in Ergebnisdateien gespeichert.",
)
use_litellm = st.sidebar.checkbox("Qwen via LiteLLM", value=False)
litellm_models = st.sidebar.text_input(
    "LiteLLM-Modell-ID(s)",
    DEFAULT_QWEN_MODEL,
    help="Mehrere über den Proxy verfügbare Modell-IDs durch Komma trennen.",
)
litellm_base_url = st.sidebar.text_input(
    "LiteLLM Base-URL",
    DEFAULT_LITELLM_BASE_URL,
    help="Responses-API-Endpunkt einschließlich /v1.",
).strip().rstrip("/")
litellm_key = st.sidebar.text_input(
    "LiteLLM API-Key",
    value=os.getenv("LITELLM_API_KEY", ""),
    type="password",
    help="Wird nur im laufenden Prozess verwendet und nie in Ergebnisdateien gespeichert.",
)
litellm_max_concurrent_requests = int(
    st.sidebar.number_input(
        "Qwen: maximal gleichzeitige Requests",
        min_value=1,
        max_value=8,
        value=1,
        step=1,
        disabled=not use_litellm,
        help=(
            "Bei 1 wird der nächste Qwen-Request garantiert erst gestartet, nachdem die vorherige Antwort vollständig "
            "eingetroffen und gespeichert ist."
        ),
    )
)
if use_litellm and litellm_max_concurrent_requests == 1:
    st.sidebar.caption("Qwen läuft strikt sequenziell: immer nur ein aktiver Request.")
if use_litellm and litellm_base_url.casefold().startswith("http://"):
    st.sidebar.warning(
        "Diese LiteLLM-URL verwendet unverschlüsseltes HTTP. API-Key und Bilddaten können auf dem Transportweg "
        "mitgelesen werden. Für Forschungsdaten ist HTTPS dringend empfohlen."
    )

use_jev = st.sidebar.checkbox("JEV via OpenRouter", value=False)
jev_threshold = .8  # Used only when inspecting historical Noul results.
jev_models = st.sidebar.text_input(
    "JEV-Modell-ID(s)",
    os.getenv("JEV_MODEL", "typesafe/jev-1.14"),
    help="Mehrere IDs durch Komma trennen. Eine feste Version macht Läufe reproduzierbar.",
)
openrouter_key = st.sidebar.text_input(
    "OpenRouter API-Key",
    value=os.getenv("OPENROUTER_API_KEY", ""),
    type="password",
    help="Wird nur im laufenden Prozess verwendet und nie in Ergebnisdateien gespeichert.",
)
if use_jev:
    st.sidebar.caption(
        "JEV verwendet dieselbe Klassifikationsaufgabe; die erlaubten Klassen stehen unter criteria."
    )

st.title("MMFM Lab")
st.caption("Reproduzierbare Zero-shot-Klassifikation von isometrischen Punktwolkenansichten")

overview_tab, preparation_tab, photo_check_tab, experiment_tab, results_tab, publication_tab = st.tabs(
    ["Übersicht", "S3DIS vorbereiten", "JEV Bildtest", "Experimente", "Ergebnisse", "Auswertungen"]
)

with overview_tab:
    st.subheader("Datensatzstatus")
    modelnet_error = None
    s3dis_error = None
    try:
        modelnet = cached_modelnet(str(modelnet_path))
    except Exception as exc:
        modelnet = pd.DataFrame()
        modelnet_error = str(exc)
    try:
        s3dis = cached_s3dis(str(s3dis_path))
    except Exception as exc:
        s3dis = pd.DataFrame()
        s3dis_error = str(exc)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ModelNet-Testobjekte", f"{modelnet['sample_id'].nunique():,}" if not modelnet.empty else "-")
    col2.metric("ModelNet-Klassen", f"{modelnet['label'].nunique():,}" if not modelnet.empty else "-")
    col3.metric("S3DIS-Instanzen", f"{len(s3dis):,}" if not s3dis.empty else "-")
    col4.metric("S3DIS-Klassen", f"{s3dis['label'].nunique():,}" if not s3dis.empty else "-")
    if modelnet_error:
        st.error(modelnet_error)
    if s3dis_error:
        st.error(s3dis_error)

    if not modelnet.empty:
        view_summary = (
            modelnet.groupby("view_count")
            .agg(Objekte=("sample_id", "nunique"), Klassen=("label", "nunique"))
            .reset_index()
            .rename(columns={"view_count": "Ansichten"})
        )
        st.dataframe(view_summary, hide_index=True, use_container_width=True)

    st.subheader("Funktionen und Auswertung")
    st.markdown(
        """
        <div class="paper-note">
        <b>Objekte statt Bildzeilen:</b> Eine Vorhersage zählt genau einmal, auch wenn vier Ansichten verwendet werden.<br>
        <b>Vollständige gepaarte ModelNet-Tests:</b> 1, 2, 3 und 4 Ansichten verwenden alle gemeinsamen Testobjekte – ohne Stichprobe oder Seed.<br>
        <b>Echte S3DIS-Ablation:</b> Dieselben RGB-Bilder werden einmal ohne und einmal mit AABB-Größe klassifiziert.<br>
        <b>Auditierbar:</b> Modellversion, exakter Klassenname, lokale Richtig/Falsch-Auswertung, Laufzeit, Fehler und API-Nutzung werden protokolliert.<br>
        <b>Stärkere Statistik:</b> Accuracy, 95-%-Intervall, Macro-F1, Balanced Accuracy, klassenspezifische Werte und gepaarter McNemar-Test.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Vorgesehene Versuchsmatrix")
    matrix = pd.DataFrame(
        [
            {"Datensatz": "ModelNet (33 vorhandene Klassen)", "Bedingung": "1 / 2 / 3 / 4 Ansichten", "Geometrie": "Nein", "Vergleich": "gepaart pro Objekt"},
            {"Datensatz": "S3DIS", "Bedingung": "4 echte RGB-Ansichten", "Geometrie": "Nein", "Vergleich": "Baseline"},
            {"Datensatz": "S3DIS", "Bedingung": "dieselben 4 RGB-Ansichten", "Geometrie": "AABB X/Y/Z", "Vergleich": "gepaarte Ablation"},
        ]
    )
    st.dataframe(matrix, hide_index=True, use_container_width=True)

with preparation_tab:
    st.subheader("S3DIS: RGB-Ansichten und AABB vorbereiten")
    st.write(
        "Jede Datei im Ordner `Annotations` wird als eine gelabelte Instanz behandelt. "
        "Die Bounding-Box wird aus allen Punkten berechnet; nur das Rendering darf bei sehr großen Objekten Punkte deterministisch reduzieren."
    )
    try:
        source_manifest = cached_s3dis(str(s3dis_path))
        if not source_manifest.empty:
            available_source_areas = set(source_manifest["area"].astype(str))
            if "Area_5" not in available_source_areas:
                raise ValueError(
                    "Area_5 wurde im gewählten S3DIS-Ordner nicht gefunden. "
                    f"Verfügbar: {', '.join(sorted(available_source_areas)) or 'keine'}"
                )
            source_manifest = select_s3dis_area(source_manifest, "Area_5")
    except Exception as exc:
        st.error(str(exc))
        source_manifest = pd.DataFrame()

    if not source_manifest.empty:
        prep_col1, prep_col2 = st.columns(2)
        image_size = int(prep_col1.selectbox("Bildgröße", [256, 384, 512, 768], index=2))
        max_points = int(prep_col2.number_input("Max. Renderpunkte", min_value=1_000, value=30_000, step=5_000))
        point_radius = int(st.slider("Punktradius in Pixeln", min_value=0, max_value=3, value=1))
        st.info(
            f"Vollständige Vorbereitung von Area_5: {len(source_manifest):,} Instanzen × 4 Ansichten = "
            f"{len(source_manifest) * 4:,} PNG-Dateien. Es wird keine Stichprobe gezogen."
        )
        overwrite = st.checkbox("Vorhandene Ansichten mit denselben IDs neu erzeugen", value=False)

        if st.button("S3DIS-Vorbereitung starten", type="primary"):
            progress_bar = st.progress(0.0)
            status = st.empty()

            def prep_progress(done: int, total: int, message: str) -> None:
                progress_bar.progress(done / total if total else 1.0)
                status.caption(f"{done:,}/{total:,} · {message}")

            try:
                with st.spinner("Echte RGB-Ansichten für Area_5 werden erzeugt …"):
                    prepared, manifest_path = prepare_s3dis_views(
                        s3dis_root=s3dis_path,
                        output_root=prepared_root,
                        area="Area_5",
                        image_size=image_size,
                        max_points=max_points,
                        point_radius=point_radius,
                        overwrite=overwrite,
                        progress=prep_progress,
                    )
            except Exception as exc:
                st.error(f"S3DIS-Vorbereitung konnte nicht gestartet werden: {type(exc).__name__}: {exc}")
            else:
                cached_prepared.clear()
                ok_count = int((prepared["status"] == "ok").sum()) if not prepared.empty else 0
                error_count = int((prepared["status"] != "ok").sum()) if not prepared.empty else 0
                st.success(
                    f"Area_5 vorbereitet: {ok_count:,} erfolgreich, {error_count:,} Fehler. "
                    f"RGB und AABB-Geometrie stehen im Manifest: {manifest_path}"
                )

        manifest_path = prepared_root / "s3dis_manifest.csv"
        prepared_now = prepared_frame(manifest_path)
        prepared_ok = prepared_now[prepared_now.get("status", "") == "ok"] if not prepared_now.empty else pd.DataFrame()
        if not prepared_ok.empty:
            st.subheader("Vorschau")
            example = prepared_ok.iloc[0]
            image_columns = st.columns(4)
            for index, column in enumerate(image_columns):
                column.image(str(example[f"view_{index}"]), caption=f"Ansicht {index + 1}", use_container_width=True)
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Sample": example["sample_id"],
                            "Klasse": example["label"],
                            "Breite X [m]": example["width"],
                            "Tiefe Y [m]": example["depth"],
                            "Höhe Z [m]": example["height"],
                            "Punkte": int(example["point_count"]),
                        }
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )

with photo_check_tab:
    st.subheader("JEV Klassifikation mit 1–2 Bildern")
    st.write("Wähle ein Objekt. JEV wählt anhand einer oder zweier Ansichten genau eine Klasse aus der bisherigen Klassenliste unter criteria.")
    st.caption("Dieser Bildtest wird separat gespeichert und fließt nicht in die wissenschaftlichen Runs ein. Für die vollständige Versuchsmatrix anschließend „Experimente“ verwenden.")
    preview_dataset = st.radio("Bildtest: Datensatz", ["ModelNet", "S3DIS"], horizontal=True)
    preview_records = pd.DataFrame()
    try:
        if preview_dataset == "ModelNet":
            preview_records = select_paired_modelnet(cached_modelnet(str(modelnet_path)), (1, 2, 3, 4))
        else:
            preview_records = prepared_frame(prepared_root / "s3dis_manifest.csv")
            if not preview_records.empty:
                preview_records = preview_records[preview_records["status"] == "ok"].copy()
    except Exception as exc:
        st.info(f"Bildtest: {exc}")
    if preview_records.empty:
        st.info("Für den Bildtest zuerst den Datensatzpfad setzen bzw. S3DIS-Ansichten vorbereiten.")
    else:
        preview_labels = tuple(sorted(preview_records["label"].unique()))
        preview_search = st.text_input("Bildtest: Objekt suchen", placeholder="z. B. chair oder Area_5").strip()
        object_rows = preview_records[["sample_id"]].drop_duplicates()
        if preview_search:
            object_rows = object_rows[object_rows["sample_id"].str.contains(preview_search, case=False, regex=False)]
        if object_rows.empty:
            st.info("Kein passendes Objekt gefunden.")
        else:
            preview_sample = st.selectbox("Bildtest: Objekt", object_rows["sample_id"].tolist())
            sample_rows = preview_records[preview_records["sample_id"] == preview_sample]
            preview_row = sample_rows.sort_values("view_count").iloc[-1] if "view_count" in sample_rows else sample_rows.iloc[0]
            preview_count = st.radio("Bildtest: Anzahl Bilder", [1, 2], index=1, horizontal=True)
            try:
                preview_images = image_paths_from_row(preview_row, preview_count)
            except (ValueError, OSError) as exc:
                preview_images = []
                st.warning(f"Bildtest: {exc}")
            for image_index, (column, path) in enumerate(zip(st.columns(preview_count), preview_images), start=1):
                column.image(str(path), caption=f"Ansicht {image_index}", use_container_width=True)
            preview_geometry = None
            if preview_dataset == "S3DIS" and st.checkbox("Bildtest: AABB-Maße verwenden", value=False):
                preview_geometry = {name: float(preview_row[name]) for name in ("width", "depth", "height")}
                st.write({f"{name} [m]": value for name, value in preview_geometry.items()})
            preview_models = split_models(jev_models)
            preview_model = st.selectbox("Bildtest: JEV-Modell", preview_models or ["typesafe/jev-1.14"])
            preview_prompt = build_prompt(preview_dataset.lower(), preview_labels, preview_count, preview_geometry, criteria_reference=True)
            with st.expander("Bildtest: Fragen und Kriterien ansehen"):
                st.json(JevProvider.build_questions(preview_prompt, preview_labels))
            st.caption(f"1 Anfrage mit {preview_count} Bild(ern), einer Klassenauswahl und {len(preview_labels)} erlaubten Klassen. Der API-Key wird aus der Seitenleiste verwendet.")
            preview_confirmed = st.checkbox("Ich bestätige die API-Kosten für diesen einzelnen Bildtest.", value=False)
            if st.button("1–2 Bilder mit JEV testen", type="primary", disabled=not preview_images):
                if not openrouter_key.strip():
                    st.error("OpenRouter API-Key fehlt.")
                elif not preview_confirmed:
                    st.error("Kostenbestätigung für den Bildtest fehlt.")
                else:
                    try:
                        with st.spinner("JEV prüft die Klassen anhand der Bilder …"):
                            preview_result, preview_path = run_classification_preview(
                                preview_images, preview_dataset.lower(), preview_labels, preview_model,
                                openrouter_key, runs_root.parent / "jev_photo_checks", geometry=preview_geometry,
                                sample_id=preview_sample, true_label=str(preview_row["label"]),
                            )
                        st.session_state["jev_preview_result"] = preview_result
                        st.session_state["jev_preview_path"] = str(preview_path)
                    except Exception as exc:
                        st.error(f"JEV-Bildtest fehlgeschlagen: {type(exc).__name__}: {exc}")
            if "jev_preview_result" in st.session_state:
                saved_preview = st.session_state["jev_preview_result"]
                st.caption(f"Zuletzt getestetes Objekt: {saved_preview['sample_id']} · {saved_preview['view_count']} Ansicht(en) · {saved_preview['dataset']}")
                preview_metrics = st.columns(3)
                preview_metrics[0].metric("Bildtest: Vorhersage", str(saved_preview["pred_label"]))
                preview_metrics[1].metric("Bildtest: Ground Truth", str(saved_preview["true_label"]))
                preview_usage = json.loads(str(saved_preview["usage_json"]))
                preview_metrics[2].metric("Bildtest: Kosten [USD]", str(preview_usage.get("cost", "–")))
                with st.expander("Bildtest: gespeicherte Anfrage und Rohantwort"):
                    st.json(saved_preview)
                st.caption(f"Bildtest gespeichert: {st.session_state['jev_preview_path']}")

with experiment_tab:
    st.subheader("Experiment konfigurieren")
    dataset_label = st.radio("Datensatz", ["ModelNet", "S3DIS", "ModelNet + S3DIS (alle Runs)"], horizontal=True)
    run_both = dataset_label == "ModelNet + S3DIS (alle Runs)"
    config_col1, config_col2 = st.columns(2)
    run_name = config_col1.text_input("Laufname", "alle-runs-paper-v2" if run_both else f"{dataset_label.lower()}-paper-v2")
    repeat_count = int(config_col2.number_input("Wiederholungen", min_value=1, max_value=10, value=1))
    api_col1, api_col2 = st.columns(2)
    max_retries = int(api_col1.number_input("Wiederholungen bei API-Fehlern", min_value=0, max_value=8, value=2))
    request_delay = float(api_col2.number_input("Pause zwischen Requests [s]", min_value=0.0, max_value=30.0, value=0.0, step=0.25))
    targets = configured_targets()

    experiments: list[tuple[pd.DataFrame, ExperimentSpec]] = []
    configuration_errors: list[str] = []
    if dataset_label == "ModelNet" or run_both:
        try:
            full_manifest = cached_modelnet(str(modelnet_path))
            view_counts = [1, 2, 3, 4] if run_both else st.multiselect("Anzahl Ansichten", [1, 2, 3, 4], default=[1, 2, 3, 4])
            selected_records = pd.DataFrame()
            if view_counts:
                selected_records = select_paired_modelnet(full_manifest, view_counts)
            st.caption("Es werden alle gemeinsamen ModelNet-Testobjekte der gewählten Ansichtszahlen klassifiziert.")
            labels = tuple(sorted(selected_records["label"].unique())) if not selected_records.empty else MODELNET_CLASSES
            spec = ExperimentSpec(
                dataset="modelnet",
                labels=labels,
                area_scope="all",
                view_counts=tuple(sorted(view_counts)),
                conditions=(),
                repeat_count=repeat_count,
                max_retries=max_retries,
                request_delay_s=request_delay,
                run_name=f"{run_name}-modelnet" if run_both else run_name,
            )
            experiments.append((selected_records, spec))
        except Exception as exc:
            configuration_errors.append(f"ModelNet: {exc}")
            st.error(str(exc))
    if dataset_label == "S3DIS" or run_both:
        manifest_path = prepared_root / "s3dis_manifest.csv"
        prepared = prepared_frame(manifest_path)
        if prepared.empty:
            st.warning("Zuerst im Reiter „S3DIS vorbereiten“ RGB-Ansichten erzeugen.")
            configuration_errors.append("Vorbereitete S3DIS-Ansichten fehlen.")
        else:
            prepared = prepared[prepared["status"] == "ok"].copy()
            available_areas = sorted(prepared["area"].astype(str).unique()) if "area" in prepared else []
            selected_area = st.selectbox(
                "S3DIS Area",
                ["Alle Areas", *available_areas],
                index=1 + available_areas.index("Area_5") if run_both and "Area_5" in available_areas else 0,
                help="Begrenzt den Test auf alle Instanzen einer einzelnen S3DIS-Area.",
            )
            if selected_area != "Alle Areas":
                prepared = select_s3dis_area(prepared, selected_area)
            else:
                prepared = select_s3dis_area(prepared, "all")
            condition_labels = ["RGB ohne Geometrie", "RGB + AABB-Geometrie"] if run_both else st.multiselect(
                "Ablationsbedingungen",
                ["RGB ohne Geometrie", "RGB + AABB-Geometrie"],
                default=["RGB ohne Geometrie", "RGB + AABB-Geometrie"],
            )
            conditions = tuple(
                value
                for label, value in (
                    ("RGB ohne Geometrie", "rgb"),
                    ("RGB + AABB-Geometrie", "rgb_aabb"),
                )
                if label in condition_labels
            )
            selected_records = prepared.sort_values(["label", "sample_id"]).reset_index(drop=True)
            area_caption = selected_area if selected_area != "Alle Areas" else "alle Areas"
            st.caption(f"Es werden alle erfolgreich vorbereiteten S3DIS-Instanzen für {area_caption} klassifiziert.")
            labels = tuple(sorted(selected_records["label"].unique())) if not selected_records.empty else S3DIS_CLASSES
            spec = ExperimentSpec(
                dataset="s3dis",
                labels=labels,
                area_scope=selected_area if selected_area != "Alle Areas" else "all",
                view_counts=(4,),
                conditions=conditions,
                repeat_count=repeat_count,
                max_retries=max_retries,
                request_delay_s=request_delay,
                run_name=f"{run_name}-s3dis" if run_both else run_name,
            )
            experiments.append((selected_records, spec))

    if experiments:
        requests = sum(planned_request_count(records, spec, targets) for records, spec in experiments if not records.empty)
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        summary_col1.metric("Objekte", f"{sum(records['sample_id'].nunique() for records, _ in experiments if not records.empty):,}")
        summary_col2.metric("Klassen", f"{sum(records['label'].nunique() for records, _ in experiments if not records.empty):,}")
        summary_col3.metric("Modellziele", len(targets))
        summary_col4.metric("Geplante Requests", f"{requests:,}")
        if run_both:
            st.info("Ein Start führt alle vier ModelNet-Ansichtszahlen und beide S3DIS-Bedingungen nacheinander aus. Jeder Datensatz erhält einen eigenen Ergebnisordner.")
            st.dataframe(pd.DataFrame([
                {"Datensatz": spec.dataset, "Umfang": spec.area_scope,
                 "Objekte": records["sample_id"].nunique(),
                 "Requests": planned_request_count(records, spec, targets)}
                for records, spec in experiments if not records.empty
            ]), hide_index=True, use_container_width=True)
        if targets:
            st.dataframe(pd.DataFrame([target.public() for target in targets]), hide_index=True, use_container_width=True)
        paid_targets = [target for target in targets if target.provider != "mock"]
        confirmed = st.checkbox("Ich bestätige, dass die ausgewählten API-Modelle Kosten verursachen können.", value=False, disabled=not paid_targets)
        insecure_litellm_targets = [
            target
            for target in targets
            if target.provider == "litellm" and target.base_url.casefold().startswith("http://")
        ]
        insecure_confirmed = st.checkbox(
            "Ich bestätige die Übertragung von API-Key und Bildern über die unverschlüsselte LiteLLM-HTTP-Verbindung.",
            value=False,
            disabled=not insecure_litellm_targets,
        )

        if st.button("Alle Runs starten" if run_both else "Experiment starten", type="primary"):
            problems = list(configuration_errors)
            if any(records.empty for records, _ in experiments):
                problems.append("Keine Samples für mindestens einen Datensatz ausgewählt.")
            if any(not spec.view_counts if spec.dataset == "modelnet" else not spec.conditions for _, spec in experiments):
                problems.append("Keine Versuchsbedingungen ausgewählt.")
            if not targets:
                problems.append("Kein Modell ausgewählt.")
            if paid_targets and not confirmed:
                problems.append("Kostenbestätigung fehlt.")
            if any(target.provider == "gemini" and not target.api_key for target in targets):
                problems.append("Gemini API-Key fehlt.")
            if use_jev and not split_models(jev_models):
                problems.append("JEV-Modell-ID fehlt.")
            if any(target.provider in {"jev", "jev_photo_check"} and not target.api_key.strip() for target in targets):
                problems.append("OpenRouter API-Key fehlt.")
            if any(target.provider == "litellm" and not target.api_key for target in targets):
                problems.append("LiteLLM API-Key fehlt.")
            if any(target.provider == "litellm" and not target.base_url for target in targets):
                problems.append("LiteLLM Base-URL fehlt.")
            if insecure_litellm_targets and not insecure_confirmed:
                problems.append("Bestätigung für die unverschlüsselte LiteLLM-HTTP-Verbindung fehlt.")
            if problems:
                for problem in problems:
                    st.error(problem)
            else:
                progress_bar = st.progress(0.0)
                run_status = st.empty()
                live_latest = st.empty()
                live_table = st.empty()
                live_rows: list[dict[str, object]] = []
                completed_requests = 0

                def run_progress(
                    done: int,
                    total: int,
                    message: str,
                    result: dict[str, object],
                ) -> None:
                    done = completed_requests + done
                    total = requests
                    progress_bar.progress(done / total if total else 1.0)
                    run_status.caption(f"{done:,}/{total:,} · {message}")
                    status = str(result.get("status", "error"))
                    correct = result.get("correct") is True
                    assessment = "richtig" if status == "ok" and correct else "falsch" if status == "ok" else "Fehler"
                    answer = str(result.get("pred_label", "") or "–")
                    live_row = {
                        "Nr.": done,
                        "Sample": str(result.get("sample_id", "")),
                        "Datensatz": str(result.get("dataset", "")),
                        "Modell": f"{result.get('provider', '')} / {result.get('model', '')}",
                        "Antwort": answer,
                        "Ground Truth": str(result.get("true_label", "")),
                        "Bewertung": assessment,
                        "Bedingung": str(result.get("condition", "")),
                        "Laufzeit [s]": result.get("latency_s", ""),
                    }
                    if result.get("provider") == "jev_photo_check" and status == "ok":
                        scores = photo_check_scores(str(result["raw_response"]))
                        live_row["Photo Check: P(Ja)"] = scores[answer]
                        live_row["Photo Check: prüfen"] = scores[answer] < jev_threshold or sum(value >= jev_threshold for value in scores.values()) > 1
                    live_rows.append(live_row)
                    latest_text = (
                        f"Antwort {done:,}/{total:,}: {live_row['Sample']} → {answer} "
                        f"(Ground Truth: {live_row['Ground Truth']}, {assessment})"
                    )
                    if assessment == "richtig":
                        live_latest.success(latest_text)
                    elif assessment == "falsch":
                        live_latest.warning(latest_text)
                    else:
                        error_text = str(result.get("error", "") or "Unbekannter Fehler")
                        live_latest.error(f"{latest_text} · {error_text}")
                    live_table.dataframe(
                        pd.DataFrame(live_rows[-200:]),
                        hide_index=True,
                        use_container_width=True,
                    )

                for selected_records, spec in experiments:
                    run_dir = create_run_directory(runs_root, spec, targets, selected_records)
                    st.session_state["last_run_dir"] = str(run_dir)
                    try:
                        with st.spinner(f"{spec.dataset}: Experiment läuft; jedes Ergebnis wird sofort gespeichert …"):
                            results, evaluated = run_experiment(selected_records, spec, targets, run_dir, progress=run_progress)
                    except Exception as exc:
                        st.error(f"{spec.dataset}: Lauf abgebrochen: {type(exc).__name__}: {exc}. Bereits gespeicherte Ergebnisse: {run_dir}")
                    else:
                        successes = int((results["status"] == "ok").sum())
                        errors = int((results["status"] != "ok").sum())
                        st.success(f"{spec.dataset}: {successes:,} erfolgreiche Antworten, {errors:,} Fehler. Ergebnisse: {run_dir}")
                        if not evaluated["aggregate"].empty:
                            st.dataframe(evaluated["aggregate"], hide_index=True, use_container_width=True)
                    completed_requests += planned_request_count(selected_records, spec, targets)

with results_tab:
    st.subheader("Gespeicherte Läufe")
    st.caption("Grafiken und Datenexporte findest du im Reiter Auswertungen.")
    with st.expander("Alte ModelNet-Ergebnisse einheitlich importieren", expanded=False):
        st.write(
            "Erzeugt einen unveränderten ModelNet40-Auditlauf und einen Vergleich mit exakt denselben "
            "1.860 Objekt-IDs und 33 Klassen wie die aktuellen ModelNet-Daten."
        )
        try:
            legacy_bundle = load_legacy_results(old_results_path)
            legacy_overview = legacy_bundle.profile.copy()
            st.dataframe(legacy_overview, hide_index=True, use_container_width=True)
            if st.button("Alte Resultate importieren und auswerten"):
                current_manifest = cached_modelnet(str(modelnet_path))
                imported = import_legacy_results(old_results_path, current_manifest, runs_root)
                st.success(
                    "Import abgeschlossen: vollständiger 40-Klassen-Audit und exakt angeglichener "
                    "33-Klassen-Vergleich wurden als normale Ergebnisläufe gespeichert."
                )
                st.code(str(imported["comparable33"]), language=None)
                st.rerun()
        except (LegacyResultError, OSError, ValueError, pd.errors.ParserError) as exc:
            st.warning(str(exc))
    with st.expander("Ergebnisdaten exportieren", expanded=False):
        st.write(
            "Fasst aktuelle Läufe und den exakt angeglichenen 33-Klassen-Altvergleich zusammen. "
            "Unvollständige Läufe werden separat ausgewiesen."
        )
        if st.button("Ergebnisdaten aktualisieren"):
            paper_export = build_paper_export(runs_root, DEFAULT_PAPER_EXPORT)
            st.success(f"Ergebnisexport aktualisiert: {paper_export.output_dir}")
        paper_zip = DEFAULT_PAPER_EXPORT / "MMFM_Paper_Results_CSV.zip"
        if paper_zip.exists():
            st.download_button(
                "Ergebnisse als ZIP herunterladen",
                data=paper_zip.read_bytes(),
                file_name=paper_zip.name,
                mime="application/zip",
            )
        paper_workbook = DEFAULT_PAPER_EXPORT / "MMFM_Paper_Results.xlsx"
        if paper_workbook.exists():
            st.download_button(
                "Ergebnisse als Excel herunterladen",
                data=paper_workbook.read_bytes(),
                file_name=paper_workbook.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    run_dirs = sorted(
        [path for path in runs_root.glob("*") if path.is_dir() and (path / "results.csv").exists()],
        key=lambda path: path.name,
        reverse=True,
    ) if runs_root.exists() else []
    render_confusion_explorer(run_dirs, DEFAULT_HISTORICAL_CSV, runs_root.parent / "historical_overview" / "original_pilot")
    if not run_dirs:
        st.info("Noch keine gespeicherten Experimentläufe. Historische CSV-Dateien lassen sich im Matrix-Explorer oben auswählen.")
    else:
        with st.expander("Combine runs: compare legacy and current models", expanded=False):
            comparable_run_dirs = [path for path in run_dirs if run_scope(path) != "modelnet40_legacy_full"]
            run_labels = {
                path: f"{PROTOCOL_LABELS.get(run_scope(path), run_scope(path))} · {path.name}"
                for path in comparable_run_dirs
            }
            recommended_runs: list[Path] = []
            for path in comparable_run_dirs:
                if run_scope(path) == "modelnet33_exact_ids":
                    recommended_runs.append(path)
                    continue
                repeat_summary = read_optional_csv(path / "summary_by_repeat.csv")
                if repeat_summary.empty:
                    continue
                planned = pd.to_numeric(repeat_summary.get("n_planned"), errors="coerce")
                coverage = pd.to_numeric(repeat_summary.get("coverage"), errors="coerce")
                is_modelnet = repeat_summary.get("dataset", "").astype(str).str.casefold().eq("modelnet")
                is_scientific = ~repeat_summary.get("provider", "").astype(str).str.casefold().eq("mock")
                if (is_modelnet & is_scientific & planned.eq(1860) & coverage.ge(0.95)).any():
                    recommended_runs.append(path)

            selected_comparison_runs = st.multiselect(
                "Runs included in the combined analysis",
                comparable_run_dirs,
                default=recommended_runs,
                format_func=lambda path: run_labels[path],
                help=(
                    "Complete scientific ModelNet runs and the exactly aligned legacy comparison are selected by "
                    "default. Partial or test runs can be added manually."
                ),
            )
            keep_duplicate_tasks = st.checkbox(
                "Keep identical tasks from different runs as additional repetitions",
                value=False,
                help=(
                    "When disabled, identical task IDs are counted once. Enable this only when the same tasks were "
                    "intentionally repeated as independent measurements."
                ),
            )

            if selected_comparison_runs:
                signatures = tuple(
                    (
                        int((path / "results.csv").stat().st_mtime_ns),
                        int((path / "results.csv").stat().st_size),
                    )
                    for path in selected_comparison_runs
                )
                merged_results = cached_merged_run_results(
                    tuple(str(path) for path in selected_comparison_runs),
                    signatures,
                    keep_duplicate_tasks,
                )
                merged_summary = summarize_merged_results(merged_results)
                modelnet_summary = merged_summary[
                    merged_summary["dataset"].astype(str).str.casefold().eq("modelnet")
                ].copy()

                if modelnet_summary.empty:
                    st.info("The selected runs contain no ModelNet results for the 1–4 view comparison.")
                else:
                    protocol_options = sorted(modelnet_summary["protocol"].astype(str).unique())
                    model_options = sorted(modelnet_summary["model"].astype(str).unique())
                    recommended_models = sorted(
                        modelnet_summary.loc[
                            ~modelnet_summary["provider"].astype(str).str.casefold().eq("mock"),
                            "model",
                        ].unique()
                    )
                    filter_col1, filter_col2 = st.columns(2)
                    selected_protocols = filter_col1.multiselect(
                        "Protocols",
                        protocol_options,
                        default=protocol_options,
                    )
                    selected_comparison_models = filter_col2.multiselect(
                        "Models",
                        model_options,
                        default=recommended_models or model_options,
                    )
                    metric_labels = {
                        "end_to_end_accuracy": "End-to-end Accuracy",
                        "accuracy_valid_only": "Valid-response Accuracy",
                        "coverage": "Coverage",
                    }
                    selected_comparison_metric = st.selectbox(
                        "Comparison metric",
                        list(metric_labels),
                        format_func=lambda value: metric_labels[value],
                    )
                    comparison_rows = modelnet_summary[
                        modelnet_summary["protocol"].isin(selected_protocols)
                        & modelnet_summary["model"].isin(selected_comparison_models)
                    ].copy()
                    comparison_rows = comparison_rows.sort_values(
                        ["protocol", "provider", "model", "view_count"]
                    )

                    if comparison_rows.empty:
                        st.info("No results are available for the selected protocols and models.")
                    else:
                        merged_object_count = int(
                            merged_results.loc[
                                merged_results["dataset"].astype(str).str.casefold().eq("modelnet"), "sample_id"
                            ].nunique()
                        )
                        overview_columns = st.columns(4)
                        overview_columns[0].metric("Selected runs", len(selected_comparison_runs))
                        overview_columns[1].metric("Unique objects", f"{merged_object_count:,}")
                        overview_columns[2].metric("Models", comparison_rows["model"].nunique())
                        overview_columns[3].metric(
                            "Available views",
                            ", ".join(
                                str(int(value)) for value in sorted(comparison_rows["view_count"].dropna().unique())
                            ),
                        )

                        view_table = modelnet_view_table(comparison_rows, selected_comparison_metric)
                        percentage_columns = [
                            column
                            for column in ("1 view", "2 views", "3 views", "4 views")
                            if column in view_table
                        ]
                        styled_view_table = view_table.style.format(
                            {column: "{:.1%}" for column in percentage_columns},
                            na_rep="N/A",
                        )
                        st.dataframe(styled_view_table, hide_index=True, width="stretch")
                        st.caption(
                            "N/A means that no valid class-name response exists. A measured 0.0% end-to-end "
                            "accuracy or coverage remains a valid zero result."
                        )

                        comparison_figure = px.line(
                            comparison_rows,
                            x="view_count",
                            y=selected_comparison_metric,
                            color="model",
                            line_dash="protocol",
                            line_group="comparison_series",
                            markers=True,
                            hover_data={
                                "protocol": True,
                                "n_objects": True,
                                "n_classes": True,
                                "n_runs": True,
                                "n_planned": True,
                                "view_count": False,
                                "comparison_series": False,
                                "model": False,
                            },
                            labels={
                                "view_count": "Number of isometric views",
                                selected_comparison_metric: metric_labels[selected_comparison_metric],
                                "model": "Model",
                                "protocol": "Protocol",
                                "n_objects": "Objects",
                                "n_classes": "Classes",
                                "n_runs": "Runs",
                                "n_planned": "Evaluated responses",
                            },
                            title=f"ModelNet classification by number of isometric views — {metric_labels[selected_comparison_metric]}",
                        )
                        comparison_figure.update_traces(connectgaps=False)
                        comparison_figure.update_xaxes(tickmode="array", tickvals=[1, 2, 3, 4], range=[0.8, 4.2])
                        comparison_figure.update_yaxes(range=[-0.03, 1.02], tickformat=".0%", zeroline=True)
                        comparison_figure.update_layout(hovermode="x unified", template="simple_white")
                        model_only_legend(comparison_figure)
                        chart_config = {
                            "displaylogo": False,
                            "toImageButtonOptions": {
                                "format": st.session_state.get("cm_export_format", "svg"),
                                "filename": "modelnet_legacy_current_views_1_to_4",
                                "scale": 2,
                            },
                        }
                        st.plotly_chart(comparison_figure, width="stretch", config=chart_config)

                        if selected_comparison_metric != "coverage":
                            coverage_figure = px.line(
                                comparison_rows,
                                x="view_count",
                                y="coverage",
                                color="model",
                                line_dash="protocol",
                                line_group="comparison_series",
                                markers=True,
                                labels={
                                    "view_count": "Number of isometric views",
                                    "coverage": "Coverage",
                                    "model": "Model",
                                    "protocol": "Protocol",
                                },
                                title="Response coverage by number of isometric views",
                            )
                            coverage_figure.update_traces(connectgaps=False)
                            coverage_figure.update_xaxes(
                                tickmode="array", tickvals=[1, 2, 3, 4], range=[0.8, 4.2]
                            )
                            coverage_figure.update_yaxes(range=[-0.03, 1.02], tickformat=".0%", zeroline=True)
                            coverage_figure.update_layout(hovermode="x unified", template="simple_white")
                            model_only_legend(coverage_figure)
                            st.plotly_chart(coverage_figure, width="stretch", config=chart_config)

                        st.subheader("Confusion matrices by number of views")
                        confusion_col1, confusion_col2 = st.columns(2)
                        selected_confusion_model = confusion_col1.selectbox(
                            "Model for confusion matrices",
                            sorted(comparison_rows["model"].astype(str).unique()),
                            key="merged_confusion_model",
                        )
                        matrix_shared = st.session_state.get("confusion_display_settings", {}).get("modelnet")
                        normalize_confusion = matrix_shared[0].normalize if matrix_shared else True
                        confusion_col2.caption("Gemeinsame Matrixfilter oben einstellen. Die Referenz gilt auch für diese Alt-/Neu-Grafik.")
                        selected_scopes = comparison_rows.loc[
                            comparison_rows["model"].astype(str).eq(selected_confusion_model),
                            "protocol_scope",
                        ].unique()
                        confusion_results = merged_results[
                            merged_results["protocol_scope"].isin(selected_scopes)
                            & merged_results["model"].astype(str).eq(selected_confusion_model)
                        ]
                        confusion_figure = confusion_matrix_grid(
                            confusion_results,
                            model=selected_confusion_model,
                            normalize=normalize_confusion,
                            settings=matrix_shared[0] if matrix_shared else None,
                            reference=matrix_shared[1] if matrix_shared else None,
                        )
                        confusion_chart_config = {
                            "displaylogo": False,
                            "toImageButtonOptions": {
                                "format": st.session_state.get("cm_export_format", "svg"),
                                "filename": f"{selected_confusion_model}_confusion_matrices_views_1_to_4",
                                "scale": 2,
                            },
                        }
                        st.plotly_chart(confusion_figure, width="stretch", config=confusion_chart_config)

                        export_columns = [
                            "protocol",
                            "provider",
                            "model",
                            "dataset",
                            "condition",
                            "view_count",
                            "n_runs",
                            "n_planned",
                            "n_objects",
                            "n_classes",
                            "n_success",
                            "n_correct",
                            "coverage",
                            "end_to_end_accuracy",
                            "accuracy_valid_only",
                            "mean_latency_s",
                        ]
                        download_col1, download_col2, download_col3 = st.columns(3)
                        download_col1.download_button(
                            "Download combined comparison table (CSV)",
                            data=comparison_rows[export_columns].to_csv(index=False).encode("utf-8"),
                            file_name="modelnet_merged_old_new_views_1_to_4.csv",
                            mime="text/csv",
                        )
                        download_col2.download_button(
                            "Download comparison figure (interactive HTML)",
                            data=comparison_figure.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8"),
                            file_name="modelnet_old_new_views_1_to_4.html",
                            mime="text/html",
                        )
                        download_col3.download_button(
                            "Download confusion matrix panel (interactive HTML)",
                            data=confusion_figure.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8"),
                            file_name=f"{selected_confusion_model}_merged_confusion_views_1_to_4.html",
                            mime="text/html",
                        )

            else:
                st.info("Select at least one run for the combined analysis.")

        st.subheader("Inspect a single run")
        default_index = 0
        last_run = st.session_state.get("last_run_dir")
        if last_run:
            for index, path in enumerate(run_dirs):
                if str(path) == last_run:
                    default_index = index
                    break
        selected_run = st.selectbox("Run", run_dirs, index=default_index, format_func=lambda path: path.name)
        aggregate_path = selected_run / "summary_aggregate.csv"
        paired_path = selected_run / "paired_comparisons.csv"
        results_path = selected_run / "results.csv"
        aggregate = read_optional_csv(aggregate_path)
        paired = read_optional_csv(paired_path)
        raw_results = load_results_csv(results_path)

        if not aggregate.empty:
            st.dataframe(aggregate, hide_index=True, use_container_width=True)
            single_metric_labels = {
                "end_to_end_accuracy_mean": "End-to-end Accuracy",
                "accuracy_mean": "Valid-response Accuracy",
                "macro_f1_mean": "Macro F1",
                "balanced_accuracy_mean": "Balanced Accuracy",
                "coverage_mean": "Coverage",
            }
            metric = st.selectbox(
                "Metric",
                [
                    value
                    for value in single_metric_labels
                    if value in aggregate
                ],
                format_func=lambda value: single_metric_labels[value],
            )
            chart_data = aggregate.copy()
            chart_data["Model"] = chart_data["model"].astype(str)
            chart_data["view_count"] = pd.to_numeric(chart_data["view_count"], errors="coerce")
            is_modelnet_run = chart_data["dataset"].astype(str).str.casefold().eq("modelnet").any()
            if is_modelnet_run:
                figure = px.line(
                    chart_data[chart_data["dataset"].astype(str).str.casefold().eq("modelnet")],
                    x="view_count",
                    y=metric,
                    color="Model",
                    markers=True,
                    labels={
                        "view_count": "Number of isometric views",
                        metric: single_metric_labels[metric],
                    },
                    title=f"{single_metric_labels[metric]} by number of isometric views",
                )
                figure.update_traces(connectgaps=False)
                figure.update_xaxes(tickmode="array", tickvals=[1, 2, 3, 4], range=[0.8, 4.2])
            else:
                chart_data["Condition"] = chart_data["condition"].astype(str)
                figure = px.bar(
                    chart_data,
                    x="Condition",
                    y=metric,
                    color="Model",
                    barmode="group",
                    labels={metric: single_metric_labels[metric]},
                    title=f"{single_metric_labels[metric]} by experimental condition",
                )
            figure.update_yaxes(range=[-0.03, 1.02], tickformat=".0%", zeroline=True)
            figure.update_layout(legend_title_text="Model", template="simple_white")
            single_chart_config = {
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": st.session_state.get("cm_export_format", "svg"),
                    "filename": f"{selected_run.name}_{metric}",
                    "scale": 2,
                },
            }
            st.plotly_chart(figure, width="stretch", config=single_chart_config)
            st.download_button(
                "Download single-run figure (interactive HTML)",
                data=figure.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8"),
                file_name=f"{selected_run.name}_{metric}.html",
                mime="text/html",
            )

        if not paired.empty:
            st.subheader("Paired comparisons")
            st.dataframe(paired, hide_index=True, use_container_width=True)

        single_modelnet_results = raw_results[
            raw_results["dataset"].astype(str).str.casefold().eq("modelnet")
        ].copy() if not raw_results.empty else pd.DataFrame()
        if not single_modelnet_results.empty:
            st.subheader("Confusion matrices by number of views")
            single_confusion_col1, single_confusion_col2 = st.columns(2)
            single_confusion_models = sorted(single_modelnet_results["model"].astype(str).unique())
            selected_single_confusion_model = single_confusion_col1.selectbox(
                "Model for single-run confusion matrices",
                single_confusion_models,
            )
            matrix_shared = st.session_state.get("confusion_display_settings", {}).get("modelnet")
            single_confusion_normalized = matrix_shared[0].normalize if matrix_shared else True
            single_confusion_col2.caption("Gleiche Darstellung wie im gemeinsamen Matrix-Explorer; gilt auch für importierte Altresultate.")
            single_confusion_figure = confusion_matrix_grid(
                single_modelnet_results,
                model=selected_single_confusion_model,
                normalize=single_confusion_normalized,
                settings=matrix_shared[0] if matrix_shared else None,
                reference=matrix_shared[1] if matrix_shared else None,
            )
            single_confusion_config = {
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": st.session_state.get("cm_export_format", "svg"),
                    "filename": f"{selected_single_confusion_model}_single_run_confusion_views_1_to_4",
                    "scale": 2,
                },
            }
            st.plotly_chart(single_confusion_figure, width="stretch", config=single_confusion_config)
            st.download_button(
                "Download confusion matrix panel (interactive HTML)",
                data=single_confusion_figure.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8"),
                file_name=f"{selected_single_confusion_model}_confusion_views_1_to_4.html",
                mime="text/html",
            )

        if not raw_results.empty:
            st.subheader("Einzelergebnisse")
            result_view = raw_results.copy()
            successful_mask = result_view["status"].astype(str) == "ok"
            locally_correct = (
                result_view["true_label"].astype(str).str.casefold()
                == result_view["pred_label"].astype(str).str.casefold()
            )
            result_view["Bewertung"] = "Fehler"
            result_view.loc[successful_mask & locally_correct, "Bewertung"] = "richtig"
            result_view.loc[successful_mask & ~locally_correct, "Bewertung"] = "falsch"
            result_view["Modell"] = result_view["provider"].astype(str) + " / " + result_view["model"].astype(str)
            result_view["Ergebnistyp"] = "API-Modell"
            result_view.loc[result_view["provider"].astype(str).str.casefold() == "mock", "Ergebnistyp"] = (
                "Pipeline-Test (Mock)"
            )
            result_view["Ansichten"] = pd.to_numeric(result_view["view_count"], errors="coerce").fillna(0).astype(int)

            model_rows = result_view[["Modell", "provider"]].drop_duplicates()
            scientific_models = sorted(
                model_rows.loc[model_rows["provider"].astype(str).str.casefold() != "mock", "Modell"].tolist()
            )
            mock_models = sorted(
                model_rows.loc[model_rows["provider"].astype(str).str.casefold() == "mock", "Modell"].tolist()
            )
            model_options = [*scientific_models, *mock_models, "Alle Modelle"]
            selected_result_model = st.selectbox(
                "Angezeigtes Modell",
                model_options,
                index=0,
                help="Echte API-Modelle werden standardmäßig getrennt vom kostenfreien Mock-Pipeline-Test angezeigt.",
            )
            displayed_results = result_view.copy()
            if selected_result_model != "Alle Modelle":
                displayed_results = displayed_results[displayed_results["Modell"] == selected_result_model].copy()

            contains_mock = (displayed_results["provider"].astype(str).str.casefold() == "mock").any()
            contains_scientific = (displayed_results["provider"].astype(str).str.casefold() != "mock").any()
            if contains_mock and contains_scientific:
                st.warning(
                    "Diese Auswahl mischt API-Ergebnisse mit dem künstlichen Mock-Pipeline-Test. "
                    "Für die wissenschaftliche Auswertung bitte ein einzelnes API-Modell auswählen."
                )
            elif contains_mock:
                st.info(
                    "Angezeigt wird nur der kostenfreie Mock-Pipeline-Test. Seine Klassen sind künstliche Testwerte "
                    "und dürfen nicht als Modellergebnis interpretiert werden."
                )

            result_metrics = st.columns(3)
            result_metrics[0].metric("Richtig", f"{int((displayed_results['Bewertung'] == 'richtig').sum()):,}")
            result_metrics[1].metric("Falsch", f"{int((displayed_results['Bewertung'] == 'falsch').sum()):,}")
            result_metrics[2].metric("Fehler", f"{int((displayed_results['Bewertung'] == 'Fehler').sum()):,}")
            st.dataframe(
                displayed_results[
                    [
                        "sample_id",
                        "Modell",
                        "Ergebnistyp",
                        "true_label",
                        "pred_label",
                        "Bewertung",
                        "condition",
                        "Ansichten",
                        "repeat",
                        "latency_s",
                    ]
                ].rename(
                    columns={
                        "sample_id": "Sample",
                        "true_label": "Ground Truth",
                        "pred_label": "Vorhersage",
                        "condition": "Bedingung",
                        "repeat": "Wiederholung",
                        "latency_s": "Laufzeit [s]",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

            with st.expander("Einzelergebnis und verwendete Isometrien anzeigen"):
                condition_options = sorted(displayed_results["condition"].astype(str).unique())
                selected_condition = st.selectbox("Bedingung filtern", ["Alle", *condition_options])
                detail_rows = displayed_results.copy()
                if selected_condition != "Alle":
                    detail_rows = detail_rows[detail_rows["condition"].astype(str) == selected_condition]

                search_text = st.text_input("Sample suchen", placeholder="z. B. chair oder Area_5").strip().casefold()
                if search_text:
                    detail_rows = detail_rows[
                        detail_rows["sample_id"].astype(str).str.casefold().str.contains(search_text, regex=False)
                    ]
                if detail_rows.empty:
                    st.info("Für diesen Filter wurde kein Einzelergebnis gefunden.")
                else:
                    if len(detail_rows) > 1_000:
                        st.caption(
                            f"Der Detailwähler zeigt die ersten 1.000 von {len(detail_rows):,} Treffern. "
                            "Die vollständige Tabelle steht oben; für ein bestimmtes Objekt bitte Suche oder Filter verwenden."
                        )
                        detail_rows = detail_rows.head(1_000)
                    detail_indices = detail_rows.index.tolist()

                    def detail_label(index: int) -> str:
                        row = result_view.loc[index]
                        return (
                            f"{row['sample_id']} · {row['true_label']} → {row['pred_label'] or 'Fehler'} "
                            f"· {row['Bewertung']} · {row['Modell']} · {row['condition']}"
                        )

                    selected_index = st.selectbox("Ergebnis", detail_indices, format_func=detail_label)
                    detail = result_view.loc[selected_index]
                    info_columns = st.columns(4)
                    info_columns[0].metric("Ground Truth", str(detail["true_label"]))
                    info_columns[1].metric("Vorhersage", str(detail["pred_label"] or "–"))
                    info_columns[2].metric("Bewertung", str(detail["Bewertung"]))
                    info_columns[3].metric("Ansichten", int(detail["Ansichten"]))

                    try:
                        image_paths = [Path(path) for path in json.loads(str(detail["image_paths_json"]))]
                    except (TypeError, ValueError, json.JSONDecodeError):
                        image_paths = []
                    available_images = [path for path in image_paths if path.is_file()]
                    if available_images:
                        image_columns = st.columns(len(available_images))
                        for image_index, (column, image_path) in enumerate(zip(image_columns, available_images), start=1):
                            column.image(str(image_path), caption=f"Isometrie {image_index}", use_container_width=True)
                    else:
                        st.warning("Die für dieses Ergebnis gespeicherten Bilddateien wurden nicht gefunden.")
                    if str(detail.get("error", "")):
                        st.error(str(detail["error"]))
                    show_photo_check(detail, jev_threshold, "result-photo-check-csv")

        errors = raw_results[raw_results["status"] != "ok"] if not raw_results.empty else pd.DataFrame()
        if not errors.empty:
            with st.expander(f"Fehler ({len(errors):,})"):
                st.dataframe(errors[["sample_id", "provider", "model", "condition", "error"]], hide_index=True, use_container_width=True)

        download_col1, download_col2 = st.columns(2)
        photo_checks_path = selected_run / "photo_checks.csv"
        if photo_checks_path.exists():
            st.download_button("Alle Photo-Check-Werte als CSV", photo_checks_path.read_bytes(),
                               file_name=f"{selected_run.name}_photo_checks.csv", mime="text/csv")
        download_col1.download_button(
            "Rohdaten als CSV",
            data=results_path.read_bytes(),
            file_name=f"{selected_run.name}_results.csv",
            mime="text/csv",
        )
        if aggregate_path.exists():
            download_col2.download_button(
                "Zusammenfassung als CSV",
                data=aggregate_path.read_bytes(),
                file_name=f"{selected_run.name}_summary.csv",
                mime="text/csv",
            )


with publication_tab:
    st.subheader("Auswertungen")
    st.write("ModelNet-Detailmatrizen, vollständige Matrizen, S3DIS-Klassenvergleich und gepaarte AABB-Wechsel, "
             "Jev-Wahrscheinlichkeiten sowie die Pilot-Abbildung aus gespeicherten Ergebnissen erstellen.")
    render_publication_export(
        runs_root,
        DEFAULT_PAPER_EXPORT.parent / "auswertungen",
        reference_figures=find_paper_reference_figures(PROJECT_DIR),
        historical_csv_root=DEFAULT_HISTORICAL_CSV,
    )
