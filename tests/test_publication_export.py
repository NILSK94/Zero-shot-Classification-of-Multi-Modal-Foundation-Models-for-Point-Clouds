from pathlib import Path
import json
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from mmfm_tool.publication_export import (load_publication_rows,select_panels,focus_matrices,
    panel_matrix,panel_metrics,build_publication_export)


def example_rows():
    rows=[]
    for model in ['a','b']:
        for view in [1,4]:
            for i,truth in enumerate(['chair','chair','table','table']):
                prediction=truth if (i<2 or view==4) else 'chair'
                rows.append(dict(task_id=f'{model}-{view}-{i}',timestamp='2026-10-20T10:00:00Z',
                    provider='test',model=model,dataset='modelnet',sample_id=f'object{i}',true_label=truth,
                    pred_label=prediction,condition=f'views_{view}',view_count=view,repeat=1,status='ok'))
    return pd.DataFrame(rows)


def profile():
    return {'version':'test','modelnet_objects':4,'modelnet_classes':2,'s3dis_objects':4,
            's3dis_classes':2,'minimum_coverage':0,'single_view_models':[],'s3dis_models':[],
            'overview_order':['a','b'],'models':[
                {'model':m,'provider':'test','scope':'current_paper_protocol','title':m.upper(),
                 'stem':m.upper(),'key':m,'color':color,'views':[1,4],
                 'gains':['table'],'difficult':['chair']} for m,color in [('a','#d62728'),('b','#1f77b4')]]}


class PublicationExportTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]/'tmp'; root.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=root); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.runs=self.root/'runs'; self.runs.mkdir()

    def save_run(self,name,rows,scope='current_paper_protocol'):
        folder=self.runs/name; folder.mkdir()
        rows.to_csv(folder/'results.csv',index=False)
        (folder/'config.json').write_text(json.dumps({'scope':scope}),encoding='utf-8')
        return folder

    def test_success_precedes_newer_failed_resume_and_duplicate_copy(self):
        rows=example_rows(); self.save_run('original',rows)
        failed=rows.iloc[[0]].copy(); failed['timestamp']='2026-10-21T00:00:00Z'; failed['status']='error'
        failed['task_id']='different-task-id'; failed['pred_label']=''
        self.save_run('later_failure',failed)
        self.save_run('copy',rows)
        data,inventory=load_publication_rows(self.runs)
        self.assertEqual(len(data),16)
        self.assertTrue(data.loc[data.sample_id.eq('object0'),'_valid'].all())
        self.assertEqual(len(inventory),3)

    def test_protocols_never_collapse_and_truth_conflicts_are_reported(self):
        rows=example_rows(); self.save_run('current',rows)
        self.save_run('legacy',rows,'other_protocol')
        data,_=load_publication_rows(self.runs); self.assertEqual(len(data),32)
        bad=rows.iloc[[0]].copy(); bad['true_label']='wrong'; self.save_run('bad',bad)
        with self.assertRaisesRegex(ValueError,'Ground-Truth'): load_publication_rows(self.runs)

    def test_exact_ids_are_required_not_only_matching_counts(self):
        rows=example_rows(); rows.loc[rows.model.eq('b'),'sample_id']+='different'
        self.save_run('objects',rows); data,_=load_publication_rows(self.runs)
        models,_,_,excluded=select_panels(data,profile())
        self.assertEqual([m['definition']['model'] for m in models],['a'])
        self.assertTrue(any('Objekt-IDs' in x['reason'] for x in excluded))

    def test_other_predictions_and_failures_preserve_the_original_denominator(self):
        rows=example_rows().query("model == 'a'").copy()
        rows.loc[(rows.view_count.eq(4))&(rows.sample_id.eq('object2')),['pred_label','status']]=['','error']
        rows.loc[(rows.view_count.eq(4))&(rows.sample_id.eq('object3')),'pred_label']='outside'
        self.save_run('decisions',rows); data,_=load_publication_rows(self.runs)
        models,_,_,_=select_panels(data,profile()); item=models[0]
        focus=focus_matrices(item['panels'],item['definition'],['chair','table'])
        m=focus['matrices'][4]
        self.assertEqual(m['support']['table'],2)
        self.assertEqual(m['counts'].loc['table','Other / invalid'],1)
        self.assertEqual(m['counts'].loc['table','outside'],1)
        self.assertTrue(np.allclose(m['shares'].sum(axis=1),1))
        self.assertEqual(panel_metrics(item['panels'][4])['accuracy'],.5)
        self.assertNotIn('Other / invalid',m['counts'].index)
        self.assertEqual(m['counts'].loc['table','table'],0)

    def test_repetitions_do_not_pool_and_missing_conditions_are_not_fabricated(self):
        rows=example_rows(); self.save_run('repeat1',rows)
        self.save_run('repeat2',rows.assign(repeat=2))
        data,_=load_publication_rows(self.runs)
        models,_,_,_=select_panels(data,profile(),2)
        self.assertEqual(sum(len(p) for m in models for p in m['panels'].values()),16)
        rows=rows[~(rows.model.eq('b')&rows.view_count.eq(4))]
        self.save_run('partial_only',rows.assign(repeat=3))
        models,_,_,excluded=select_panels(load_publication_rows(self.runs)[0],profile(),3)
        self.assertEqual(len(models),1)
        self.assertTrue(any('letzte' in e['reason'] for e in excluded))

    def test_complete_latex_package_contains_all_referenced_files_and_metadata(self):
        self.save_run('data',example_rows())
        path=self.root/'profile.json'; path.write_text(json.dumps(profile()))
        result=build_publication_export(self.runs,self.root/'export',profile_path=path,png_dpi=60)
        self.assertTrue(result.zip_path.is_file())
        import re
        tex='\n'.join(p.read_text(encoding='utf-8') for p in (result.output_dir/'paper_tex').rglob('*.tex'))
        for image in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',tex):
            self.assertTrue((result.output_dir/image).is_file(),image)
        self.assertIn('Figure~\\ref{fig:a_full_confusion}',tex)
        self.assertEqual(result.manifest['appendix_order'][0]['figure'],'A1')
        self.assertEqual(result.manifest['class_overview_figure'],'A3')
        with zipfile.ZipFile(result.zip_path) as archive:
            self.assertIn('PAPER_FIGURES_GUIDE.md',archive.namelist())
            self.assertIn('paper_tex/appendix_modelnet.tex',archive.namelist())
            self.assertTrue(all(not Path(p).is_absolute() and '..' not in Path(p).parts for p in archive.namelist()))
        self.assertNotIn('raw_response',(result.output_dir/'manifest.json').read_text())

    def test_changed_rankings_do_not_claim_the_profile_is_still_the_largest_gain(self):
        rows=example_rows().query("model == 'a'").copy()
        # Swap the gains: chair now improves; table becomes worse.
        rows.loc[rows.view_count.eq(1),'pred_label']='table'
        rows.loc[rows.view_count.eq(4),'pred_label']='chair'
        self.save_run('changed',rows); data,_=load_publication_rows(self.runs)
        models,_,_,_=select_panels(data,profile())
        focus=focus_matrices(models[0]['panels'],models[0]['definition'],['chair','table'])
        self.assertFalse(focus['approved_gains_still_largest'])

    def test_complete_package_includes_s3dis_hard_decision_figure_and_demo_input(self):
        rows=example_rows(); self.save_run('modelnet',rows)
        s3=rows[rows.view_count.eq(4)].copy(); s3['dataset']='s3dis'
        rgb=s3.assign(condition='rgb'); aabb=s3.assign(condition='rgb_aabb')
        self.save_run('s3dis',pd.concat([rgb,aabb],ignore_index=True))
        custom=profile(); custom['s3dis_models']=['a','b']
        path=self.root/'profile.json'; path.write_text(json.dumps(custom))
        result=build_publication_export(self.runs,self.root/'export',profile_path=path,png_dpi=60)
        self.assertEqual(result.manifest['s3dis_confusions']['models'],['a','b'])
        self.assertTrue((result.output_dir/'paper_figures/s3dis_confusion_rgb_aabb_4models.pdf').is_file())
        self.assertIn(r'\input{paper_tex/s3dis_confusion}',(result.output_dir/'figures_demo.tex').read_text())
        self.assertTrue((result.output_dir/'paper_data/S3DIS_object_predictions.csv').is_file())
        paired=result.manifest['s3dis_paired_analysis']
        self.assertEqual(paired['objects_per_model'],4)
        self.assertEqual(paired['models'],['a','b'])
        self.assertTrue((result.output_dir/paired['paired_figure']).is_file())
        demo=(result.output_dir/'figures_demo.tex').read_text()
        self.assertIn(r'\input{paper_tex/s3dis_paired_results}',demo)
        self.assertIn(r'\input{paper_tex/appendix_s3dis}',demo)
        self.assertIn(r'B\arabic{figure}',(result.output_dir/'paper_tex/appendix_s3dis.tex').read_text())

    def test_optional_pilot_archive_is_exported_with_a_reproducible_caption_replay(self):
        project=Path(__file__).resolve().parents[1]
        archive=project.parent/'Multimodal_old/Multimodal-point-cloud-classifier-main/data_results_csv'
        if not archive.exists():self.skipTest('Local historical archive is not installed.')
        self.save_run('modelnet',example_rows())
        path=self.root/'profile.json'; path.write_text(json.dumps(profile()))
        result=build_publication_export(self.runs,self.root/'export',profile_path=path,png_dpi=60,historical_csv_root=archive)
        self.assertEqual(len(result.manifest['pilot_figure7']['panels']),12)
        self.assertTrue((result.output_dir/'paper_figures/Pilot_Figure7_confusion_matrices.png').is_file())
        self.assertTrue((result.output_dir/'paper_tex/pilot_confusion.tex').is_file())
        self.assertIn(r'\input{paper_tex/pilot_confusion}',(result.output_dir/'figures_demo.tex').read_text())
        self.assertIn('ground-truth-dependent',result.manifest['pilot_figure7']['stable_diffusion']['method'])


if __name__=='__main__': unittest.main()
