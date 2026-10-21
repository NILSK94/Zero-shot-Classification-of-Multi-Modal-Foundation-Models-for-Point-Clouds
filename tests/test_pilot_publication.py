from pathlib import Path
import tempfile
import unittest
import csv
import hashlib
import numpy as np
from mmfm_tool.pilot_publication import pilot_panels,replay_caption_evaluation


class PilotPublicationTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[1]/'tmp'; base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.evaluator=self.root/'evaluator.py'
        self.evaluator.write_text("SYNONYMS={'airplane':['plane'], 'chair':['chair']}\nraise RuntimeError('This code must never execute')\n")

    def test_caption_rule_is_explicitly_ground_truth_dependent_and_never_executes_script(self):
        source=self.root/'stable.csv'
        source.write_text('filename,prompt\nairplane_0001_iso_0.png,a plane\nchair_0002_iso_0.png,a plane\n')
        records,info=replay_caption_evaluation(source,self.evaluator)
        self.assertEqual(records.pred_label.tolist(),['airplane','unknown'])
        self.assertIn('ground-truth-dependent',info['method'])
        self.assertEqual(info['repaired_outer_quoting_rows'],0)

    def test_outer_quoting_repair_preserves_fields_and_source_bytes(self):
        source=self.root/'stable.csv'
        with source.open('w',newline='') as stream:
            writer=csv.writer(stream); writer.writerow(['filename','prompt'])
            writer.writerow(['airplane_0001_iso_0.png,"a plane, black and white"'])
        before=hashlib.sha256(source.read_bytes()).hexdigest()
        records,info=replay_caption_evaluation(source,self.evaluator)
        self.assertEqual(records.pred_label.tolist(),['airplane'])
        self.assertEqual(info['repaired_outer_quoting_rows'],1)
        self.assertEqual(before,hashlib.sha256(source.read_bytes()).hexdigest())

    def test_ambiguous_or_missing_synonym_dictionary_is_rejected(self):
        source=self.root/'stable.csv'; source.write_text('filename,prompt\nairplane_01.png,a plane\n')
        self.evaluator.write_text('SYNONYMS={}\nSYNONYMS={}\n')
        with self.assertRaises(ValueError):replay_caption_evaluation(source,self.evaluator)

    def test_actual_archive_keeps_all_12_panels_and_exact_counts(self):
        project=Path(__file__).resolve().parents[1]
        archive=project.parent/'Multimodal_old/Multimodal-point-cloud-classifier-main/data_results_csv'
        if not archive.exists():self.skipTest('Local historical archive is not installed.')
        panels,replay=pilot_panels(archive,project/'Evaluator_stablediffusion.py')
        self.assertEqual(len(panels),12)
        for p in panels:
            self.assertEqual(int(p['counts'].to_numpy().sum()),len(p['data']))
            self.assertEqual(int(np.trace(p['counts'].to_numpy())),int(p['data'].true_label.eq(p['data'].pred_label).sum()))
        self.assertEqual([len(p['data']) for p in panels[:4]],[363,726,1089,1452])
        self.assertEqual([len(p['data']) for p in panels[4:8]],[363,726,1089,1452])
        self.assertEqual([len(p['data']) for p in panels[8:]],[1860,1860,35,35])
        self.assertEqual(int(np.trace(panels[8]['counts'].to_numpy())),322)
        self.assertEqual(int(np.trace(panels[11]['counts'].to_numpy())),23)
        self.assertEqual(replay['repaired_outer_quoting_rows'],1860)


if __name__=='__main__':unittest.main()
