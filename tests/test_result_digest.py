from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mmfm_tool.confusion_analysis import MatrixSettings, make_reference, matrix_values, prepare_confusion_rows
from mmfm_tool.historical_csv import load_historical_csvs
from mmfm_tool.result_digest import automatic_digest


def example_results():
    rows = []
    for model in ("a", "b", "partial"):
        for view in (1, 4):
            for i in range(1 if model == "partial" else 10):
                truth = "good" if i < 5 else "bad"
                correct = i < 5 or (model == "a" and view == 4 and i < 7) or (model == "b" and view == 1 and i == 5)
                rows.append(dict(dataset="modelnet", provider="test", model=model, condition=f"views_{view}",
                                 view_count=view, repeat=1, protocol_scope="current", sample_id=str(i),
                                 true_label=truth, pred_label=truth if correct else "good",
                                 status="error" if model == "a" and view == 1 and i == 9 else "ok",
                                 task_id=f"{model}-{view}-{i}"))
    return pd.DataFrame(rows)


class DigestTests(unittest.TestCase):
    def test_shared_cohort_failures_partial_series_and_paired_changes(self):
        data = prepare_confusion_rows(example_results(), "modelnet", paired=False)
        digest = automatic_digest(data)
        self.assertEqual(digest["excluded_series"], 1)
        self.assertEqual(digest["ranking"].iloc[0]["model_label"], "test / a")
        self.assertEqual(digest["ranking"].iloc[0]["accuracy"], .7)
        first = digest["summary"].query("model_label == 'test / a' and view_count == 1").iloc[0]
        self.assertEqual(first["accuracy"], .5)
        self.assertEqual(first["coverage"], .9)
        best_gain = digest["gains"].iloc[0]
        self.assertEqual((best_gain["gains"], best_gain["losses"], best_gain["paired_observations"]), (2, 0, 10))
        self.assertEqual(digest["difficult"]["label"].tolist(), ["bad"])
        self.assertEqual(digest["stable"]["label"].tolist(), ["good"])

    def test_never_pool_protocols_and_do_not_hide_rare_classes(self):
        raw = example_results()
        rare = raw[raw["sample_id"].eq("0") & raw["model"].ne("partial")].copy()
        rare["sample_id"], rare["true_label"], rare["pred_label"] = "rare-id", "rare", "rare"
        rare["task_id"] += "-rare"
        data = prepare_confusion_rows(pd.concat([raw, rare]), "modelnet", paired=False)
        digest = automatic_digest(data)
        self.assertNotIn("rare", digest["stable"]["label"].tolist())
        shown = sorted(set(digest["profiles"]["label"]) - set(digest["stable"]["label"]))
        settings = MatrixSettings(focus="manual")
        reference = make_reference(digest["data"], settings, shown)
        reference["other_name"] = "* stable classes"
        matrix = matrix_values(digest["detail"], reference, settings)
        self.assertIn("rare", matrix["counts"].index)
        self.assertEqual(matrix["counts"].to_numpy().sum(), len(digest["detail"]))
        self.assertIn("* stable classes", matrix["names"].values())
        data.loc[0, "protocol_scope"] = "different"
        with self.assertRaises(ValueError):
            automatic_digest(data)

    def test_original_pilot_matches_old_notebook_rows_and_label_extraction(self):
        root = Path(__file__).resolve().parents[1] / "tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            source = pd.DataFrame({"image": ["flower_pot_001_iso_0.png", "flower_pot_001_iso_1.png", "chair_002_iso_0.png", "chair_002_iso_1.png"],
                                   "label": ["flowerpot", "flowerpot", "chair", "table"]})
            source.to_csv(Path(folder) / "GPT-4o-2-Isometric-views.csv", index=False)
            original, audit = load_historical_csvs(folder, original_rows=True)
            self.assertEqual(len(original), 4)
            self.assertEqual(original["sample_id"].nunique(), 2)
            self.assertEqual(audit.iloc[0]["source_rows"], 4)
            expected = pd.crosstab(source["image"].str.split("_").str[0].str.lower(), source["label"].str.strip().str.lower())
            actual = pd.crosstab(original["true_label"], original["pred_label"])
            pd.testing.assert_frame_equal(actual, expected, check_names=False)
            objects, _ = load_historical_csvs(folder)
            self.assertEqual(len(objects), 2)
            self.assertEqual(objects.loc[objects["true_label"].eq("chair"), "status"].iloc[0], "error")

    def test_default_view_has_no_cluster_controls(self):
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parents[1] / "tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            run = Path(folder) / "run"
            run.mkdir()
            example_results().to_csv(run / "results.csv", index=False)
            source = ("from pathlib import Path\nfrom mmfm_tool.confusion_ui import render_confusion_explorer\n"
                      f"render_confusion_explorer([Path({str(run)!r})], Path({str(Path(folder) / 'missing')!r}))")
            app = AppTest.from_string(source, default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Bestes Ergebnis" in element.value for element in app.markdown))
            self.assertTrue(any("Ausgefiltert" in element.value for element in app.caption))
            self.assertFalse(any("Cluster" in element.label or "Klassenfokus" == element.label for element in app.selectbox))


if __name__ == "__main__":
    unittest.main()
