from pathlib import Path
import hashlib,json,tempfile,unittest,zipfile
from streamlit.testing.v1 import AppTest
from mmfm_tool.evaluation_export import build_evaluation_export
from test_publication_export import example_rows,profile


class EvaluationExportTests(unittest.TestCase):
    def test_neutral_bundle_keeps_figure_names_data_and_provenance_but_excludes_manuscript_files(self):
        base=Path(__file__).resolve().parents[1]/'tmp';base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as folder:
            root=Path(folder);run=root/'runs/saved';run.mkdir(parents=True)
            example_rows().to_csv(run/'results.csv',index=False)
            (run/'config.json').write_text(json.dumps({'scope':'current_paper_protocol'}))
            config=root/'profile.json';config.write_text(json.dumps(profile()))
            result=build_evaluation_export(root/'runs',root/'evaluation',profile_path=config,png_dpi=60)
            self.assertEqual(result.zip_path.name,'MMFM_Auswertungen.zip')
            self.assertEqual(result.manifest['export_kind'],'evaluation')
            self.assertTrue((result.output_dir/'figures/A_confusion_focus.png').is_file())
            self.assertNotIn('appendix_order',result.manifest)
            for relative,digest in result.manifest['files'].items():
                self.assertEqual(hashlib.sha256((result.output_dir/relative).read_bytes()).hexdigest(),digest)
            with zipfile.ZipFile(result.zip_path) as archive:
                self.assertIsNone(archive.testzip())
                self.assertTrue(any(name.startswith('data/') and name.endswith('.csv') for name in archive.namelist()))
                self.assertFalse(any(name.endswith(('.tex','.md')) or 'paper_tex/' in name for name in archive.namelist()))
            app=AppTest.from_string(f'''
from pathlib import Path
from mmfm_tool.publication_ui import render_publication_export
render_publication_export(Path('not-used'),Path({str(result.output_dir)!r}))
''').run(timeout=20)
            self.assertEqual(len(app.exception),0,[e.message for e in app.exception])
            self.assertIn('A_confusion_focus.png',app.selectbox(key='publication_single_file').options)
            self.assertFalse(any('LaTeX' in str(x.label) for x in app.button))


if __name__=='__main__':unittest.main()
