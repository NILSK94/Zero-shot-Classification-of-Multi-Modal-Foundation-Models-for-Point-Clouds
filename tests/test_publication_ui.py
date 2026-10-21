from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest


class PublicationUiTests(unittest.TestCase):
    def test_export_panel_loads_without_requesting_credentials_or_running_models(self):
        app=AppTest.from_string('''
from pathlib import Path
from mmfm_tool.publication_ui import render_publication_export
render_publication_export(Path('does-not-exist'),Path('tmp/ui_no_export'))
''').run(timeout=15)
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any('Auswertungen erzeugen' in b.label for b in app.button))
        app.button(key='publication_generate').click().run(timeout=15)
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any('Keine gespeicherten' in e.value for e in app.error))


if __name__=='__main__': unittest.main()
