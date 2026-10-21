import json
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from mmfm_tool.jev_probability_analysis import parse_choice,calibration_bins,probability_matrix,collect_probability_panels


def response(probabilities,choice='a',confidence=.11):
    return json.dumps({'answers':{'classification':{'type':'choice','choice':choice,
        'probabilities':probabilities,'confidence':confidence}}})


class JevProbabilityTests(unittest.TestCase):
    def test_rounded_mass_is_preserved_and_api_confidence_is_not_used(self):
        parsed=parse_choice(response({'a':.6,'b':.39}),['a','b'],'a')
        self.assertAlmostEqual(parsed['probability_sum'],.99)
        self.assertAlmostEqual(parsed['values'][0],.6)
        self.assertAlmostEqual(parsed['p_chosen'],.6)
        self.assertAlmostEqual(parsed['api_confidence'],.11)

    def test_incomplete_or_independent_or_invalid_vectors_are_rejected(self):
        for raw in [response({'a':1}),response({'a':.6,'b':.3}),response({'a':float('nan'),'b':.5}),
                    response({'a':True,'b':False}),response({'a':1.1,'b':-.1}),
                    json.dumps({'answers':{'classification':{'type':'boolean','probability_yes':.5}}})]:
            with self.subTest(raw=raw),self.assertRaises(ValueError): parse_choice(raw,['a','b'],'a')
        with self.assertRaises(ValueError): parse_choice(response({'a':.6,'b':.4}),['a','b'],'b')

    def test_choice_is_not_silently_replaced_by_argmax(self):
        parsed=parse_choice(response({'a':.3,'b':.7}),['a','b'],'a')
        self.assertFalse(parsed['choice_is_argmax'])
        self.assertEqual(parsed['p_chosen'],.3)
        self.assertEqual(parsed['p_max'],.7)
        self.assertTrue(parse_choice(response({'a':.5,'b':.5}),['a','b'],'a')['choice_is_argmax'])

    def test_bins_include_one_and_keep_empty_bins_and_correct_denominators(self):
        frame=pd.DataFrame({'p_chosen':[0,.1,.99,1],'is_correct':[False,True,False,True]})
        bins=calibration_bins(frame)
        self.assertEqual(bins.objects.sum(),4)
        self.assertEqual(bins.iloc[-1].objects,2)
        self.assertEqual(bins.iloc[-1].accuracy,.5)
        self.assertEqual(bins.iloc[2].objects,0)
        self.assertTrue(np.isnan(bins.iloc[2].accuracy))

    def test_probability_matrix_uses_probabilities_and_keeps_omitted_mass(self):
        frame=pd.DataFrame({'true_label':['a','a','b'],'p::a':[.6,.2,.1],'p::b':[.3,.7,.8],'p::c':[.09,.1,.1]})
        matrix=probability_matrix(frame,['a','b','c'],rows=['a','b'],columns=['a','b'])
        self.assertAlmostEqual(matrix.loc['a','a'],.4)
        self.assertAlmostEqual(matrix.loc['a','Other classes'],.095)
        self.assertAlmostEqual(matrix.loc['a'].sum(),.995)
        self.assertEqual(list(matrix.index),['a','b'])

    def test_exact_source_rows_and_probability_exclusions_have_auditable_coverage(self):
        from mmfm_tool.publication_export import load_publication_rows
        root=Path(__file__).resolve().parents[1]/'tmp'; root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            runs=Path(directory); run=runs/'saved'; run.mkdir()
            rows=[]
            for index,truth in enumerate(['a','b']):
                rows.append(dict(task_id=str(index),timestamp='2026-10-20T10:00:00Z',provider='jev',model='typesafe/jev-1.14',
                    dataset='modelnet',sample_id=str(index),true_label=truth,pred_label=truth,condition='views_1',view_count=1,
                    repeat=1,status='ok',raw_response=response({'a':.6,'b':.39}) if index==0 else '{}'))
            pd.DataFrame(rows).to_csv(run/'results.csv',index=False)
            (run/'config.json').write_text(json.dumps({'scope':'current_paper_protocol'}))
            selected,_=load_publication_rows(runs)
            item={'definition':{'provider':'jev','model':'typesafe/jev-1.14'},'focus':{},'panels':{1:selected}}
            panels,summary=collect_probability_panels(runs,[item],[])
            self.assertEqual(summary[0]['coverage'],.5)
            self.assertEqual(summary[0]['excluded'],1)
            self.assertIsNone(summary[0]['mean_p_chosen_incorrect'])
            self.assertEqual(panels[0]['panels'][1]['excluded'][0]['sample_id'],'1')
            self.assertEqual(panels[0]['panels'][1]['frame'].sample_id.tolist(),['0'])


if __name__=='__main__': unittest.main()
