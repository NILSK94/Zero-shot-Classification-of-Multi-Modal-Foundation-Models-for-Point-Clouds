from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from mmfm_tool.confusion_analysis import (
    INVALID, OTHER, MatrixSettings, class_profiles, confusion_clustering, confusion_figure,
    make_reference, matrix_export, matrix_values, paired_changes, prepare_confusion_rows,
)
from mmfm_tool.confusion_ui import _validate_preset
from mmfm_tool.evaluation import evaluate_results
from mmfm_tool.figures import confusion_matrix_grid
from mmfm_tool.historical_csv import load_historical_csvs


def observations():
    rows = []
    # Same recalls for A in both panels, but object predictions swap: A is not stable.
    for view, predictions in [(1, ["a", "b", "b", "b", "a"]), (4, ["b", "a", "b", "b", "a"])]:
        for index, (truth, pred) in enumerate(zip(["a", "a", "b", "b", "rare"], predictions)):
            rows.append({"dataset": "modelnet", "model": "model", "provider": "test", "condition": f"views_{view}",
                         "view_count": view, "repeat": 1, "sample_id": f"object_{index}", "true_label": truth,
                         "pred_label": pred, "status": "ok", "task_id": f"{view}-{index}",
                         "timestamp": "2026-10-20T10:00:00Z", "latency_s": .1, "protocol_scope": "current_paper_protocol",
                         "image_paths_json": "[]", "error": ""})
    return pd.DataFrame(rows)


class ConfusionTests(unittest.TestCase):
    def setUp(self):
        self.raw = observations()
        self.data = prepare_confusion_rows(self.raw, "modelnet")
        self.settings = MatrixSettings(min_support=2)

    def test_stability_uses_objects_not_equal_aggregate_recall(self):
        profile = class_profiles(self.data).set_index("label")
        self.assertEqual(profile.loc["a", "recall_range"], 0.)
        self.assertEqual(profile.loc["a", "unchanged_fraction"], 0.)
        self.assertEqual(profile.loc["b", "unchanged_fraction"], 1.)
        reference = make_reference(self.data, replace(self.settings, focus="compact"))
        self.assertIn("a", reference["shown"])
        self.assertNotIn("b", reference["shown"])
        self.assertIn("rare", reference["shown"])
        repeated = pd.concat([self.raw.assign(repeat=repeat, task_id=self.raw["task_id"] + f"r{repeat}") for repeat in (1, 2, 3)])
        repeated_profile = class_profiles(prepare_confusion_rows(repeated, "modelnet")).set_index("label")
        self.assertEqual(repeated_profile.loc["rare", "min_support"], 1)

    def test_placeholder_preserves_denominators_and_distinguishes_original_errors(self):
        settings = replace(self.settings, focus="manual")
        reference = make_reference(self.data, settings, ["b"])
        panel = self.data[self.data["view_count"] == 1]
        matrix = matrix_values(panel, reference, settings)
        self.assertEqual(matrix["counts"].to_numpy().sum(), len(panel))
        self.assertEqual(matrix["counts"].loc[OTHER, OTHER], 2)
        self.assertEqual(matrix["correct"].loc[OTHER, OTHER], 1)
        errors = matrix_values(panel, reference, replace(settings, errors_only=True))
        self.assertEqual(errors["values"].loc[OTHER, OTHER], 1 / 3)
        exported = matrix_export(panel, reference, settings)
        self.assertEqual(exported["count"].sum(), 5)
        self.assertEqual(exported["original_errors"].sum(), 2)

    def test_invalid_responses_are_visible_and_valid_only_is_explicit(self):
        raw = self.raw.copy()
        raw.loc[0, ["pred_label", "status"]] = ["", "error"]
        data = prepare_confusion_rows(raw, "modelnet")
        reference = make_reference(data, self.settings)
        panel = data[data["view_count"] == 1]
        matrix = matrix_values(panel, reference, self.settings)
        self.assertEqual(matrix["counts"].loc["a", INVALID], 1)
        self.assertEqual(matrix["values"].loc["a", INVALID], .5)
        valid = matrix_values(panel, reference, replace(self.settings, include_invalid=False))
        self.assertNotIn(INVALID, valid["counts"].columns)
        self.assertEqual(valid["support"].loc["a"], 1)

    def test_zero_support_is_missing_and_display_mask_does_not_change_counts(self):
        reference = make_reference(self.data, self.settings)
        panel = self.data[self.data["true_label"] != "rare"]
        matrix = matrix_values(panel, reference, replace(self.settings, mask_below=.6))
        self.assertTrue(matrix["displayed"].loc["rare"].isna().all())
        self.assertTrue(np.isnan(matrix["displayed"].loc["a", "a"]))
        self.assertEqual(matrix["values"].loc["a", "a"], .5)
        self.assertEqual(matrix["counts"].to_numpy().sum(), len(panel))

    def test_clustering_is_deterministic_and_grouping_conserves_all_rows(self):
        labels = ["a", "b", "rare"]
        order, groups = confusion_clustering(self.data, labels, 2)
        self.assertEqual((order, groups), confusion_clustering(self.data, list(reversed(labels)), 2))
        settings = replace(self.settings, grouping="custom")
        ref = make_reference(self.data, settings, custom_groups={"a": "AB", "b": "AB", "rare": "rare"})
        matrix = matrix_values(self.data, ref, settings)
        self.assertEqual(matrix["counts"].to_numpy().sum(), len(self.data))
        self.assertGreater(matrix["counts"].loc["AB", "AB"], matrix["correct"].loc["AB", "AB"])

    def test_protocols_and_providers_remain_separate(self):
        old = self.raw.assign(protocol_scope="modelnet33_exact_ids")
        other = self.raw.assign(provider="another")
        data = prepare_confusion_rows(pd.concat([self.raw, old, other]), "modelnet")
        self.assertEqual(data["series_id"].nunique(), 3)
        self.assertEqual(data["panel_id"].nunique(), 6)
        figure = confusion_figure(data, make_reference(data, self.settings), self.settings)
        self.assertEqual(len(figure.data), 6)
        self.assertEqual(figure.layout.coloraxis.cmax, 1.)

    def test_paired_changes_match_ids_and_repetitions_and_count_losses(self):
        changes = paired_changes(self.data).set_index("label")
        self.assertEqual(changes.loc["a", "gains"], 1)
        self.assertEqual(changes.loc["a", "losses"], 1)
        self.assertEqual(changes.loc["a", "delta_recall"], 0.)
        extra = self.raw.iloc[:1].assign(sample_id="not_paired")
        paired = prepare_confusion_rows(pd.concat([self.raw, extra]), "modelnet")
        self.assertNotIn("not_paired", paired["sample_id"].tolist())

    def test_shared_reference_is_used_for_old_graphics_and_missing_views(self):
        settings = replace(self.settings, focus="manual")
        reference = make_reference(self.data, settings, ["b"])
        figure = confusion_matrix_grid(self.raw, model="model", settings=settings, reference=reference)
        self.assertEqual(len(figure.data), 2)
        self.assertTrue(any(annotation.text == "Not evaluated" for annotation in figure.layout.annotations))
        self.assertEqual(tuple(figure.data[0].y), tuple(figure.data[1].y))
        _validate_preset(json.loads(json.dumps(reference)))

    def test_explorer_controls_and_reference_stay_consistent_in_app(self):
        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parents[1]
        (root / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root / "tmp") as folder:
            folder = Path(folder)
            app_file = folder / "app.py"
            app_file.write_text((root / "app.py").read_text(encoding="utf-8"), encoding="utf-8")
            run = folder / "artifacts" / "runs" / "current"
            run.mkdir(parents=True)
            self.raw.to_csv(run / "results.csv", index=False)
            evaluate_results(self.raw, run)
            (run / "config.json").write_text('{}', encoding="utf-8")
            environment = {"MODELNET_PATH": str(folder / "missing"), "S3DIS_PATH": str(folder / "missing"),
                           "LEGACY_RESULTS_PATH": str(folder / "missing")}
            def widget(items, label):
                return next(item for item in items if item.label == label)
            with patch.dict("os.environ", environment):
                app = AppTest.from_file(str(app_file), default_timeout=30).run()
                self.assertFalse(app.exception)
                widget(app.radio, "Auswertung").set_value("Expertenansicht")
                app.run()
                widget(app.checkbox, "Matrix-Explorer aktivieren").check()
                app.run()
                self.assertFalse(app.exception)
                widget(app.selectbox, "Klassenfokus").set_value("compact")
                widget(app.number_input, "Mindestzahl je Klasse und Panel").set_value(2)
                app.run()
                self.assertFalse(app.exception)
                saved = app.session_state["cm_reference_modelnet"]
                self.assertEqual(set(saved["shown"]), {"a", "rare"})
                widget(app.selectbox, "Gemeinsame Klassenreihenfolge").set_value("cluster")
                widget(app.selectbox, "Klassen gruppieren").set_value("cluster")
                widget(app.checkbox, "Nur Verwechslungen zeigen").check()
                app.run()
                self.assertFalse(app.exception)
                self.assertIn("modelnet", app.session_state["confusion_display_settings"])

    def test_original_paper_csvs_collapse_views_and_never_guess_caption_predictions(self):
        root = Path(__file__).resolve().parents[1] / "tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            folder = Path(folder)
            pd.DataFrame({"image": ["flower_pot_001_iso_0.png", "flower_pot_001_iso_1.png",
                                     "chair_002_iso_0.png", "chair_002_iso_1.png"],
                          "label": ["flower_pot", "flower_pot", "chair", "table"]}).to_csv(folder / "GPT-4o-2-Isometric-views.csv", index=False)
            pd.DataFrame({"filename": ["chair_003.png"], "prompt": ["a chair or a bench"]}).to_csv(folder / "captions.csv", index=False)
            frame, audit = load_historical_csvs(folder)
            self.assertEqual(len(frame), 2)
            flower = frame[frame["true_label"] == "flower_pot"].iloc[0]
            self.assertEqual(flower["sample_id"], "flower_pot/flower_pot_001")
            self.assertEqual(flower["historical_row_count"], 2)
            self.assertEqual(frame.loc[frame["true_label"] == "chair", "status"].iloc[0], "error")
            self.assertEqual(audit.loc[audit["file"] == "captions.csv", "state"].iloc[0], "not reconstructable")


if __name__ == "__main__":
    unittest.main()
