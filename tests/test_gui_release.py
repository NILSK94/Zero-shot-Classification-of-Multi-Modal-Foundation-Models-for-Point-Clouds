from pathlib import Path
import tempfile,unittest
from streamlit.testing.v1 import AppTest
from mmfm_tool.local_paths import find_paper_reference_figures
from mmfm_tool.publication_export import load_profile


class GuiReleaseTests(unittest.TestCase):
    def test_release_contains_the_required_paper_profile(self):
        profile=load_profile()
        self.assertEqual(len(profile['models']),7)
        self.assertIn('s3dis_paired_analysis',profile)

    def test_reference_detection_prefers_current_assets_and_allows_clean_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);project=root/'app';project.mkdir()
            self.assertIsNone(find_paper_reference_figures(project))
            old=root/'Workmode_paper/figures';old.mkdir(parents=True);(old/'old.png').touch()
            self.assertEqual(find_paper_reference_figures(project),old)
            current=root/'Workmode_paper/paper_figures/reference';current.mkdir(parents=True);(current/'workflow.png').touch()
            self.assertEqual(find_paper_reference_figures(project),current)

    def test_gui_starts_without_local_data_or_saved_exports_and_shows_the_publication_tab(self):
        root=Path(__file__).resolve().parents[1]
        app=AppTest.from_file(str(root/'app.py')).run(timeout=60)
        self.assertEqual(len(app.exception),0,[e.message for e in app.exception])
        self.assertIn('Auswertungen',[tab.label for tab in app.tabs])
        self.assertTrue(any('Auswertungen erzeugen'==button.label for button in app.button))


if __name__=='__main__':unittest.main()
