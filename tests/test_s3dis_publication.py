from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from mmfm_tool.s3dis_publication import decision_matrix,validate_panels,write_s3dis_confusions


def item():
    panel=pd.DataFrame({'sample_id':['01','02','03'],'true_label':['chair','chair','table'],
        'pred_label':['chair','table','table'],'_valid':[True,True,True],
        'status':['ok']*3,'view_count':[4]*3,'repeat':[1]*3,'source_run':['saved']*3})
    return {'definition':{'model':'typesafe/jev-1.14','title':'Jev-1.14'},
            'panels':{'rgb':panel.copy(),'rgb_aabb':panel.copy()}}


class S3disPublicationTests(unittest.TestCase):
    def test_counts_saved_labels_and_retains_invalid_denominators(self):
        panel=item()['panels']['rgb']; panel.loc[1,'_valid']=False
        # A probability argmax must not override the saved label.
        panel['p_chair']=1; panel['p_table']=0
        counts,support,shares=decision_matrix(panel,['chair','table'])
        self.assertEqual(counts.loc['chair','[No valid response]'],1)
        self.assertEqual(shares.loc['chair','chair'],.5)
        self.assertEqual(shares.loc['table','table'],1)
        np.testing.assert_array_equal(counts.sum(axis=1),support)

    def test_requires_paired_ids_truth_and_one_decision_per_object(self):
        valid=item(); self.assertEqual(validate_panels([valid]),['chair','table'])
        changed=item(); changed['panels']['rgb_aabb'].loc[0,'sample_id']='other'
        with self.assertRaises(ValueError): validate_panels([changed])
        changed=item(); changed['panels']['rgb_aabb'].loc[0,'true_label']='table'
        with self.assertRaises(ValueError): validate_panels([changed])
        changed=item(); changed['panels']['rgb_aabb'].loc[0,'sample_id']='02'
        with self.assertRaises(ValueError): validate_panels([changed])
        changed=item(); del changed['panels']['rgb_aabb']
        with self.assertRaises(ValueError): validate_panels([changed])

    def test_export_pairs_ground_truth_and_both_decisions_and_emits_matching_latex(self):
        base=Path(__file__).resolve().parents[1]/'tmp'; base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            root=Path(directory); data=item(); data['panels']['rgb_aabb'].loc[1,'pred_label']='chair'
            manifest=write_s3dis_confusions(root,[data],dpi=60)
            paired=pd.read_csv(root/'paper_data/Jev_S3DIS_objektweise_RGB_RGB_AABB.csv',dtype={'sample_id':str})
            self.assertEqual(len(paired),3)
            self.assertEqual(paired.set_index('sample_id').loc['02','pred_rgb'],'table')
            self.assertEqual(paired.set_index('sample_id').loc['02','pred_rgb_aabb'],'chair')
            audit=pd.read_csv(root/'paper_data/S3DIS_confusion_validation.csv')
            self.assertEqual(audit.correct.tolist(),[2,3])
            self.assertEqual(manifest['objects_per_condition'],3)
            self.assertTrue((root/manifest['figure']).is_file())
            self.assertIn('fig:s3dis_confusion',(root/'paper_tex/s3dis_confusion.tex').read_text())


if __name__=='__main__':unittest.main()
