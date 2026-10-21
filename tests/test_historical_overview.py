from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image

from mmfm_tool.confusion_analysis import MatrixSettings
from mmfm_tool.historical_csv import load_historical_csvs
from mmfm_tool.historical_overview import (
    historical_overview_figure, overview_panels, overview_values, save_historical_overview,
)


def make_archive(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for model, token in [("Gemini-1.5-pro", "Iso.-views"), ("GPT-4o", "Isometric-views")]:
        for view in range(1, 5):
            pd.DataFrame({"image": [f"chair_0001_iso_{i}.png" for i in range(view)] +
                                     [f"table_0002_iso_{i}.png" for i in range(view)],
                          "label": ["chair"] * view + ["chair"] * view}).to_csv(folder / f"{model}-{view}-{token}.csv", index=False)
    pd.DataFrame({"image": ["chair_0001_iso_0.png", "table_0002_iso_0.png", "chair_0003_iso_0.png"],
                  "label": ["chair", "table", "table"]}).to_csv(folder / "Gemini-1.5-pro-1-Iso.-view-big-test.csv", index=False)
    for model in ("Gemini-1.5-pro", "GPT-4o"):
        for variant in ("colored-cloud", "feature-geometries"):
            pd.DataFrame({"image": ["chair_01"], "label": ["chair"]}).to_csv(folder / f"{model}-4-Isometric-views-{variant}.csv", index=False)
    pd.DataFrame({"filename": ["chair_0001.png"], "prompt": ["a chair or table"]}).to_csv(folder / "Stable-diffusion-caption.csv", index=False)


class HistoricalOverviewTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / "tmp"
        test_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=test_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive_root = self.root / "data_results_csv"
        make_archive(self.archive_root)
        self.archive, self.audit = load_historical_csvs(self.archive_root)
        self.settings = MatrixSettings(min_support=1)

    def test_layout_and_independent_cohorts_preserve_object_counts(self):
        panels, refs = overview_panels(self.archive, self.settings)
        self.assertEqual([(p["row"], p["col"]) for p in panels], [(r, c) for r in (1, 2, 3) for c in (1, 2, 3, 4)])
        self.assertTrue(all(p["objects"] == 2 for p in panels[:8]))
        self.assertNotIn("matrix", panels[8])
        self.assertEqual(panels[9]["objects"], 3)
        self.assertEqual([p["objects"] for p in panels[10:]], [1, 1])
        self.assertEqual(overview_values(panels)["count"].sum(), 21)
        self.assertEqual(set(refs), {"modelnet", "blk2go"})
        self.assertEqual(list(panels[0]["matrix"]["counts"].index), list(panels[9]["matrix"]["counts"].index))

    def test_figure_uses_twelve_positions_and_honest_placeholder(self):
        panels, refs = overview_panels(self.archive, self.settings, real_model="gpt-4o")
        figure = historical_overview_figure(panels, refs, show_values=False)
        self.assertEqual(len(figure.data), 11)
        self.assertEqual(figure.data[0].type, "heatmap")
        self.assertTrue(any("Not reconstructed" in item.text for item in figure.layout.annotations))
        self.assertEqual(panels[10]["model"], "gpt-4o")
        self.assertEqual(figure.layout.coloraxis.cmax, 1.)
        self.assertEqual(figure.layout.coloraxis2.cmax, 1.)

    def test_svg_png_and_data_package(self):
        panels, refs = overview_panels(self.archive, self.settings)
        figure = historical_overview_figure(panels, refs)
        paths = save_historical_overview(self.root / "export", panels, refs, figure)
        paths = save_historical_overview(self.root / "export", panels, refs, figure)
        self.assertTrue(all(path.is_file() and path.stat().st_size > 100 for path in paths.values()))
        with Image.open(paths["png"]) as image:
            self.assertEqual(image.size, (5000, 4000))
        manifest = json.loads(paths["json"].read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["panels"]), 12)
        self.assertEqual(manifest["panels"][9]["objects"], 3)
        self.assertTrue(manifest["panels"][9]["source_files"])
        self.assertEqual(pd.read_csv(paths["csv"])["count"].sum(), 21)

    def test_historical_layout_in_app_without_normal_runs(self):
        from streamlit.testing.v1 import AppTest

        source = Path(__file__).resolve().parents[1] / "app.py"
        app_file = self.root / "app.py"
        app_file.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        def widget(items, label):
            return next(item for item in items if item.label == label)
        with patch.dict("os.environ", {"MODELNET_PATH": str(self.root / "missing"), "S3DIS_PATH": str(self.root / "missing"), "LEGACY_RESULTS_PATH": str(self.root / "missing")}):
            app = AppTest.from_file(str(app_file), default_timeout=30).run()
            self.assertFalse(app.exception)
            widget(app.radio, "Auswertung").set_value("Historische Gesamtgrafik")
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(any(item.label == "Historisches Grafikpaket mit SVG und PNG erzeugen" for item in app.button))
            widget(app.selectbox, "BLK2GO-Modell in der unteren Reihe").set_value("gpt-4o")
            app.run()
            self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
