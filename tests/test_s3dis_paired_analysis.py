from pathlib import Path
import tempfile,unittest
import pandas as pd
from mmfm_tool.s3dis_paired_analysis import paired_decisions,paired_summary,write_s3dis_paired_analysis,CHANGE_STEM,CLASS_STEM,STATES


def example():
    rgb=pd.DataFrame({'sample_id':['a','b','c','d'],'true_label':['chair']*4,
        'pred_label':['chair','table','chair','table'],'_valid':[True]*4,'status':['ok']*4})
    aabb=rgb.copy();aabb['pred_label']=['chair','chair','table','sofa']
    return {'definition':{'model':'example','title':'Example'},'panels':{'rgb':rgb,'rgb_aabb':aabb.iloc[::-1]}}


class PairedAnalysisTests(unittest.TestCase):
    def test_pairs_by_id_and_partitions_all_four_states_even_when_rows_are_reordered(self):
        pairs=paired_decisions([example()]);s=paired_summary(pairs).iloc[0]
        self.assertEqual([s[x] for x in STATES],[1,1,1,1])
        self.assertEqual(s.net_correct,0)
        self.assertEqual(s.incorrect_to_correct_percent,25)
        self.assertEqual(s.incorrect_to_incorrect_changed,1)
        self.assertEqual(s.prediction_changed,3)
        self.assertEqual(pairs.set_index('sample_id').loc['b','pred_label_rgb_aabb'],'chair')

    def test_invalid_choice_is_incorrect_even_if_label_matches_and_probability_argmax_differs(self):
        item=example();item['panels']['rgb_aabb'].loc[0,'_valid']=False
        item['panels']['rgb_aabb']['p_chair']=1.0
        pairs=paired_decisions([item]);s=paired_summary(pairs,False).iloc[0]
        self.assertFalse(pairs.set_index('sample_id').loc['a','correct_rgb_aabb'])
        self.assertEqual(s.correct_to_incorrect,2)
        self.assertEqual(s.net_correct,-1)
        self.assertEqual(s.net_correct_percent,-25)

    def test_rejects_mismatched_truth_and_duplicate_ids(self):
        item=example();item['panels']['rgb_aabb'].loc[0,'true_label']='table'
        with self.assertRaises(ValueError):paired_decisions([item])
        item=example();item['panels']['rgb_aabb'].loc[0,'sample_id']='b'
        with self.assertRaises(ValueError):paired_decisions([item])

    def test_publication_writes_paired_data_both_figures_and_cross_references(self):
        base=Path(__file__).resolve().parents[1]/'tmp';base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            root=Path(directory);result=write_s3dis_paired_analysis(root,[example()],dpi=60)
            self.assertEqual(result['objects_per_model'],4)
            for stem in [CLASS_STEM,CHANGE_STEM]:
                for suffix in ['png','pdf']:self.assertTrue((root/'paper_figures'/f'{stem}.{suffix}').is_file())
            data=pd.read_csv(root/'paper_data/S3DIS_AABB_object_pairs.csv')
            self.assertEqual(len(data),4)
            self.assertEqual(data[STATES].sum().sum(),4)
            text=(root/'paper_tex/s3dis_paired_results.tex').read_text()
            self.assertIn('1 incorrect decisions become correct',text)
            self.assertIn('1 correct decisions become incorrect',text)
            self.assertIn('app:s3dis_paired',text)


if __name__=='__main__':unittest.main()
