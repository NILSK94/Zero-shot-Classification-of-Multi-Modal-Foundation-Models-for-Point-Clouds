"""Reproducible publication figures and portable MDPI LaTeX inputs.

Only saved classification rows are read; generating figures never calls a model.
The approved profile is versioned separately from the data and copied into every
export. Full matrices retain their old horizontal view layout. Focused matrices
select true-class rows, never aggregate truth into an artificial Other diagonal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .run_comparison import run_scope
from .jev_probability_analysis import collect_probability_panels,render_probability_exports
from .s3dis_publication import write_s3dis_confusions
from .s3dis_paired_analysis import write_s3dis_paired_analysis
from .pilot_publication import save_pilot_figure

PROFILE_PATH = Path(__file__).with_name('paper_figure_profile.json')
FIELDS = {'task_id','timestamp','dataset','sample_id','true_label','pred_label',
          'provider','model','condition','view_count','repeat','status'}
KEY = ['protocol_scope','provider','model','dataset','condition','view_count','repeat','sample_id']
NOTE = 'Selected classes show the greatest improvements in recognition and classes that remain difficult to classify.'


@dataclass(frozen=True)
class PublicationExport:
    output_dir: Path
    zip_path: Path
    manifest: dict[str, Any]


def load_profile(path: str | Path | None = None) -> dict:
    profile = json.loads(Path(path or PROFILE_PATH).read_text(encoding='utf-8'))
    if not isinstance(profile.get('models'), list) or not profile['models']:
        raise ValueError('Das Grafikprofil enthält keine Modelle.')
    stems = [m['stem'] for m in profile['models']]
    if len(stems) != len(set(stems)) or any(not re.fullmatch(r'[A-Za-z0-9]+', s) for s in stems):
        raise ValueError('Die Grafiknamen im Profil müssen eindeutig und alphanumerisch sein.')
    return profile


def load_publication_rows(runs_root: str | Path) -> tuple[pd.DataFrame, list[dict]]:
    """Thin, deterministic merge; success beats a later failed resume attempt.

    Scope is part of every identity. Also deduplicate physical object decisions
    when two exports assign different task IDs to the same panel and object.
    Raw responses, prompt text, keys, and image paths are never exported.
    """
    root = Path(runs_root).expanduser().resolve()
    frames, inventory = [], []
    for folder in sorted(root.iterdir() if root.is_dir() else []):
        path = folder/'results.csv'
        if not folder.is_dir() or not path.is_file() or path.stat().st_size == 0:
            continue
        scope = run_scope(folder)
        if scope == 'modelnet40_legacy_full':
            continue
        frame = pd.read_csv(path, usecols=lambda col: col in FIELDS,dtype={'sample_id':str,'task_id':str}).fillna('')
        frame['_source_row'] = np.arange(len(frame))
        missing = (FIELDS-{'timestamp','task_id'})-set(frame.columns)
        if missing:
            raise ValueError(f'{folder.name}: fehlende Spalten: {", ".join(sorted(missing))}')
        frame['protocol_scope'] = scope
        frame['source_run'] = folder.name
        frame['timestamp'] = frame.get('timestamp', '')
        frame['task_id'] = frame.get('task_id', '')
        frame['repeat'] = pd.to_numeric(frame['repeat'], errors='raise').astype(int)
        frame['view_count'] = pd.to_numeric(frame['view_count'], errors='raise').astype(int)
        for col in ['true_label','pred_label','dataset']:
            frame[col] = frame[col].astype(str).str.strip().str.casefold()
        frame = frame[~frame.provider.astype(str).str.casefold().str.contains('mock',regex=False)]
        frame['_valid'] = frame.status.eq('ok') & frame.pred_label.ne('')
        frame['prediction'] = frame.pred_label.where(frame['_valid'], '[No valid response]')
        frame['is_correct'] = frame['_valid'] & frame.true_label.eq(frame.pred_label)
        inventory.append({'run':folder.name,'scope':scope,'rows':len(frame),
                          'results_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        frames.append(frame)
    if not frames:
        raise ValueError('Keine gespeicherten Klassifikationsergebnisse für den Grafikexport gefunden.')
    data = pd.concat(frames,ignore_index=True)
    data['_time'] = pd.to_datetime(data.timestamp,errors='coerce',utc=True)
    data = data.sort_values(['_valid','_time','source_run'],na_position='first')
    # Different truths for the same physical object are an input error, not a new sample.
    truth_key = ['protocol_scope','dataset','sample_id']
    if data.groupby(truth_key,dropna=False).true_label.nunique().gt(1).any():
        raise ValueError('Widersprüchliche Ground-Truth-Labels für dieselben Objekt-IDs.')
    data = data.drop_duplicates(KEY,keep='last').drop(columns='_time').reset_index(drop=True)
    return data,inventory


def _identity(data: pd.DataFrame, model: dict, dataset: str, repeat: int) -> pd.DataFrame:
    return data[data.protocol_scope.eq(model['scope']) & data.provider.eq(model['provider'])
                & data.model.eq(model['model']) & data.dataset.eq(dataset) & data['repeat'].eq(repeat)].copy()


def select_panels(data: pd.DataFrame, profile: dict, repeat: int = 1) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Require exact object IDs, true classes, and conditions before comparing."""
    if repeat < 1:
        raise ValueError('Die Wiederholung muss mindestens 1 sein.')
    selected, singles, s3dis, excluded = [], [], [], []
    reference_ids = reference_truth = None
    expected_objects = profile['modelnet_objects']
    expected_classes = profile['modelnet_classes']
    for definition in profile['models']:
        series = _identity(data,definition,'modelnet',repeat)
        panels = {}
        for view in definition['views']:
            panel = series[series.condition.eq(f'views_{view}') & series.view_count.eq(view)].copy()
            ids = set(panel.sample_id)
            if len(panel)!=expected_objects or len(ids)!=expected_objects or panel.true_label.nunique()!=expected_classes:
                excluded.append({'model':definition['model'],'condition':f'views_{view}',
                                 'reason':'unvollständige oder abweichende Objekt-/Klassenbasis'})
                continue
            if panel['_valid'].mean() < profile.get('minimum_coverage',.95):
                excluded.append({'model':definition['model'],'condition':f'views_{view}','reason':'zu wenige gültige Antworten'})
                continue
            truth = panel.set_index('sample_id').true_label.to_dict()
            if reference_ids is None:
                reference_ids,reference_truth = ids,truth
            if ids!=reference_ids or truth!=reference_truth:
                excluded.append({'model':definition['model'],'condition':f'views_{view}','reason':'andere Objekt-IDs oder Ground Truth'})
                continue
            panels[view] = panel
        if not panels:
            continue
        if min(definition['views']) not in panels or max(definition['views']) not in panels:
            excluded.append({'model':definition['model'],'reason':'erste oder letzte Vergleichsbedingung fehlt; keine Detailmatrix erzeugt'})
            continue
        selected.append({'definition':definition,'panels':panels})
    if not selected:
        raise ValueError('Keine vollständige Modellserie auf der im Profil festgelegten gemeinsamen Objektbasis.')
    for definition in profile.get('single_view_models',[]):
        series = _identity(data,definition,'modelnet',repeat)
        panel = series[series.condition.eq('views_1') & series.view_count.eq(1)]
        if len(panel)==expected_objects and set(panel.sample_id)==reference_ids and panel.set_index('sample_id').true_label.to_dict()==reference_truth and panel['_valid'].mean()>=profile.get('minimum_coverage',.95):
            singles.append({'definition':definition,'panel':panel})
    s3_ids = s3_truth = None
    by_model = {m['model']:m for m in profile['models']}
    for name in profile.get('s3dis_models',[]):
        definition = by_model[name]
        series = _identity(data,definition,'s3dis',repeat)
        panels = {}
        for condition in ['rgb','rgb_aabb']:
            panel = series[series.condition.eq(condition) & series.view_count.eq(4)]
            if len(panel)!=profile['s3dis_objects'] or panel.sample_id.nunique()!=profile['s3dis_objects'] or panel.true_label.nunique()!=profile['s3dis_classes']:
                continue
            ids,truth = set(panel.sample_id),panel.set_index('sample_id').true_label.to_dict()
            if s3_ids is None:
                s3_ids,s3_truth = ids,truth
            if ids==s3_ids and truth==s3_truth and panel['_valid'].mean()>=profile.get('minimum_coverage',.95):
                panels[condition] = panel
        if len(panels)==2:
            s3dis.append({'definition':definition,'panels':panels})
        else:
            excluded.append({'model':name,'dataset':'s3dis','reason':'kein vollständiger gepaarter RGB/AABB-Vergleich'})
    return selected,singles,s3dis,excluded


def panel_metrics(panel: pd.DataFrame) -> dict:
    valid = panel[panel['_valid']]
    macro = float(f1_score(valid.true_label,valid.pred_label,average='macro',zero_division=0)) if len(valid) else None
    return {'objects':int(panel.sample_id.nunique()),'observations':len(panel),
            'correct':int(panel.is_correct.sum()),'coverage':float(panel['_valid'].mean()),
            'accuracy':float(panel.is_correct.mean()),'macro_f1':macro}


def panel_matrix(panel: pd.DataFrame, labels: list[str]) -> tuple[pd.DataFrame,pd.Series,pd.DataFrame]:
    raw = pd.crosstab(panel.true_label,panel.prediction).reindex(index=labels,fill_value=0)
    support = panel.groupby('true_label').size().reindex(labels,fill_value=0)
    if support.eq(0).any():
        raise ValueError('Eine Matrix enthält Klassen ohne getestete Objekte.')
    square = raw.reindex(columns=labels,fill_value=0)
    assert int(np.trace(square.to_numpy()))==int(panel.is_correct.sum())
    return raw,support,square.div(support,axis=0)


def focus_matrices(panels: dict[int,pd.DataFrame], definition: dict, labels: list[str]) -> dict:
    first,last = min(panels),max(panels)
    selected = sorted(set(definition['gains']+definition['difficult']+definition.get('context',[])))
    if not set(selected).issubset(labels):
        raise ValueError(f"{definition['title']}: Profilklassen fehlen in der Datenbasis.")
    raw_first,support,shares_first = panel_matrix(panels[first],labels)
    raw_last,last_support,shares_last = panel_matrix(panels[last],labels)
    if not support.equals(last_support):
        raise ValueError('Die Klassenunterstützung unterscheidet sich zwischen den Bedingungen.')
    candidates = raw_first.reindex(index=selected).sum().add(raw_last.reindex(index=selected).sum(),fill_value=0).drop(selected,errors='ignore')
    # Invalid responses belong to the final collection column, never a class column.
    candidates = candidates.drop('[No valid response]',errors='ignore')
    extras = sorted(candidates[candidates.gt(0)].sort_values(ascending=False,kind='stable').head(3).index)
    columns = selected+extras
    has_invalid = any((p.true_label.isin(selected) & p.prediction.eq('[No valid response]')).any()
                      for p in [panels[first],panels[last]])
    other = 'Other / invalid' if has_invalid else 'Other classes'
    matrices = {}
    for view,raw in [(first,raw_first),(last,raw_last)]:
        counts = raw.reindex(index=selected,columns=columns,fill_value=0).copy()
        counts[other] = support.loc[selected]-counts.sum(axis=1)
        shares = counts.div(support.loc[selected],axis=0)
        assert counts.sum(axis=1).equals(support.loc[selected])
        assert np.allclose(shares.sum(axis=1),1)
        matrices[view] = {'counts':counts,'support':support.loc[selected],'shares':shares}
    gains = pd.Series(np.diag(shares_last)-np.diag(shares_first),index=labels)
    k = len(definition['gains'])
    # Ties at the selection boundary are equivalent; do not depend on sort order.
    boundary = gains.nlargest(k).iloc[-1]
    gains_match = all(gains.loc[c]>=boundary-1e-12 for c in definition['gains'])
    return {'rows':selected,'columns':columns+[other],'extras':extras,'matrices':matrices,
            'first':first,'last':last,'approved_gains_still_largest':bool(gains_match),
            'recognition_changes':{c:float(gains[c]) for c in selected}}


def _tex_escape(text: str) -> str:
    mapping = {'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}'}
    return ''.join(mapping.get(c,c) for c in text)


def _figure_tex(path: str, caption: str, label: str, *, tall: bool = False) -> str:
    size = r'width=\textwidth,height=0.82\textheight,keepaspectratio' if tall else r'width=\textwidth'
    return ('\\begin{figure*}['+('p' if tall else 'ht')+']\n\\centering\n'
            f'\\includegraphics[{size}]{{{path}}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure*}}\n')


def _write_latex(root: Path, models: list[dict], profile: dict, has_s3dis: bool) -> None:
    tex = root/'paper_tex'; (tex/'modelnet').mkdir(parents=True)
    appendix = [r'\clearpage',r'\ifdefined\appendixtitles\appendixtitles{yes}\fi',
                r'\ifdefined\appendixstart\appendixstart\fi',r'\appendix',
                r'\setcounter{figure}{0}',r'\renewcommand{\thefigure}{A\arabic{figure}}',
                r'\ifdefined\theHfigure\renewcommand{\theHfigure}{appendix.A.\arabic{figure}}\fi',
                r'\section{Complete ModelNet Confusion Matrices}',r'\label{app:complete_modelnet_confusion}',
                'Complete class-set matrices for all evaluated true classes and reported input configurations. '
                'The selected main-text classes illustrate improvements in recognition and persistent classification difficulties. '
                'Overall metrics continue to use every evaluated object.\n']
    for index,item in enumerate(models,1):
        d,f = item['definition'],item['focus']; title = _tex_escape(d['title'])
        reasons = 'the largest increases' if f['approved_gains_still_largest'] else 'selected changes'
        gains = ', '.join(_tex_escape(s.replace('_',' ')) for s in d['gains'])
        difficult = ', '.join(_tex_escape(s.replace('_',' ')) for s in d['difficult'])
        caption = (f'Selected confusion matrices of {title} with {f["first"]} and {f["last"]} input views. '
            f'The selection contrasts {reasons} in the percentage of correctly classified objects ({gains}) '
            f'with persistently difficult classes ({difficult}). Additional prediction columns retain frequent error destinations. '
            'The final column combines predictions into classes not shown individually and, where indicated, invalid responses. '
            'Original true-class denominators are retained. '
            f'Complete matrices: Appendix~\\ref{{app:complete_modelnet_confusion}}, Figure~\\ref{{fig:{d["key"]}_full_confusion}}.')
        (tex/'modelnet'/f'{d["key"]}.tex').write_text(_figure_tex(f'paper_figures/{d["stem"]}_confusion_focus.pdf',caption,f'fig:{d["key"]}_phase2_confusion'),encoding='utf-8')
        view_text = ', '.join(map(str,sorted(item['panels'])))
        caption = f'Complete class-set confusion matrices of {title} with {view_text} input views, arranged from left to right. The common blue scale shows the fraction within each true class.'
        if d['scope']!='current_paper_protocol':
            caption += ' This comparison uses the historical 40-label prompt on the evaluated 33-class object subset. Predictions outside the evaluated class subset and invalid responses remain in the denominators but are not shown as separate columns.'
        appendix.append(_figure_tex(f'paper_figures/{d["stem"]}_confusion.pdf',caption,f'fig:{d["key"]}_full_confusion'))
    appendix.extend([_figure_tex('paper_figures/ModelNet_Klassenvergleich_4Views.pdf',
        'Percentage of correctly classified objects within each evaluated class with four input views. Cells and the common blue scale show percentages.',
        'fig:modelnet_class_recall_4views',tall=True),r'\clearpage'])
    (tex/'appendix_modelnet.tex').write_text('\n'.join(appendix)+'\n',encoding='utf-8')
    (tex/'modelnet_overview.tex').write_text(_figure_tex('paper_figures/modelnet_accuracy_by_views.pdf',
        'Object-level ModelNet accuracy by number of isometric views. Missing configurations are not interpolated.',
        'fig:modelnet_accuracy_views'),encoding='utf-8')
    if has_s3dis:
        (tex/'s3dis_performance.tex').write_text(_figure_tex('paper_figures/s3dis_rgb_aabb_performance.pdf',
            'S3DIS Area 5 accuracy and Macro-F1 for RGB and paired RGB+AABB input. Values are percentages on the same objects.',
            'fig:s3dis_performance'),encoding='utf-8')
    intro = ('For readability, the main-text matrices compare the first available configuration with four views. '
             'Selected classes show the greatest improvements in recognition and classes that remain difficult to classify. '
             'Additional prediction columns show frequent error destinations; the final column combines the remaining predictions. '
             'This selection does not change the overall metrics. Complete matrices and the four-view class comparison are provided in '
             'Appendix~\\ref{app:complete_modelnet_confusion}.\n')
    (tex/'modelnet_selection_note.tex').write_text(intro,encoding='utf-8')
    includes = '\n'.join(f'\\input{{paper_tex/modelnet/{i["definition"]["key"]}}}' for i in models)
    demo = (r'\documentclass{article}'+'\n'+r'\usepackage[a4paper,margin=20mm]{geometry}'+'\n'+
            r'\usepackage{graphicx,hyperref,float}'+'\n'+r'\begin{document}'+'\n'+r'\section{ModelNet Results}'+'\n'+
            r'\input{paper_tex/modelnet_selection_note}'+'\n'+r'\input{paper_tex/modelnet_overview}'+'\n'+includes+'\n'+
            (r'\clearpage\section{S3DIS Results}\input{paper_tex/s3dis_performance}'+'\n'+r'\input{paper_tex/s3dis_paired_results}'+'\n'+r'\input{paper_tex/s3dis_confusion}'+'\n' if has_s3dis else '')+
            r'\input{paper_tex/appendix_modelnet}'+'\n'+(r'\input{paper_tex/appendix_s3dis}'+'\n' if has_s3dis else '')+r'\end{document}'+'\n')
    (root/'figures_demo.tex').write_text(demo,encoding='utf-8')


def _write_metric_tables(root: Path, models: list[dict], singles: list[dict], s3dis: list[dict]) -> None:
    tex=root/'paper_tex'
    lines=[r'\begin{table}[ht]',r'\centering',r'\small',r'\setlength{\tabcolsep}{4pt}',
        r'\caption{ModelNet object-level performance on the 33-class subset. Input is black-and-white; '
        r'Accuracy and Macro-F1 are fractions. Supplementary single-view and historical local-model comparisons are included where available.}',
        r'\label{tab:phase2_classification}',r'\begin{tabular}{lrrrrr}',r'\hline',
        r'Model & Objects & Views & Images & Accuracy & Macro-F1 \\',r'\hline']
    for item in models+singles:
        d=item['definition']; panels=item.get('panels',{1:item.get('panel')})
        for view,panel in sorted(panels.items()):
            s=panel_metrics(panel)
            lines.append(f'{_tex_escape(d["title"])} & {s["objects"]} & {view} & {s["objects"]*view} & '
                         f'{s["accuracy"]:.3f} & {s["macro_f1"]:.3f} '+r'\\')
        lines.append(r'\hline')
    lines.extend([r'\end{tabular}',r'\end{table}'])
    (tex/'modelnet_metrics_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if s3dis:
        lines=[r'\begin{table*}[ht]',r'\centering',r'\small',
            r'\caption{S3DIS Area 5 object-classification performance with four RGB views, with and without AABB dimensions. '
            r'Accuracy and Macro-F1 are fractions.}',r'\label{tab:s3dis_classification}',r'\begin{tabular}{llrrrr}',r'\hline',
            r'Model & Input & Objects & Views & Accuracy & Macro-F1 \\',r'\hline']
        for item in s3dis:
            for condition,panel in item['panels'].items():
                s=panel_metrics(panel)
                lines.append(f'{_tex_escape(item["definition"]["title"])} & {"RGB" if condition=="rgb" else "RGB+AABB"} & '
                    f'{s["objects"]} & 4 & {s["accuracy"]:.3f} & {s["macro_f1"]:.3f} '+r'\\')
            lines.append(r'\hline')
        lines.extend([r'\end{tabular}',r'\end{table*}'])
        (tex/'s3dis_metrics_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def _write_probability_latex(root: Path, figures: list[dict], stats: list[dict]) -> None:
    tex=root/'paper_tex'
    def percentage(value): return f'{value*100:.1f}\\%' if value is not None else 'not available'
    methods=(r'\paragraph{Stored class probabilities of Jev-1.14.} '+
        'In addition to the final Choice, the saved Jev responses contain a probability for every candidate class. '
        'We analyze complete, finite Choice vectors on the same evaluated objects. Values must lie in $[0,1]$ and '
        'their sum must differ from one by at most 0.01 to accommodate the precision of the saved responses. '
        'We retain these values without renormalization and report coverage and rounding in Appendix~\\ref{app:jev_probabilities}. '
        'Probability matrices show the within-true-class mean probability assigned to each candidate class; '
        'they do not count hard classification decisions. For correctness plots, we use the probability assigned to the saved Choice, '
        'rather than replacing the Choice by the largest probability. The separate API confidence field is not used. '
        'We compute a descriptive expected calibration error (ECE) in ten fixed-width bins, '
        '$\\sum_b (n_b/N)|a_b-\\bar p_b|$, where $a_b$ is observed accuracy and $\\bar p_b$ is mean probability '
        'assigned to the saved Choice; the last bin includes probability one. Empty bins contribute zero. '
        'These single-run diagnostics do not establish probability calibration. Historical independent Noul yes/no '
        'scores are excluded from this categorical analysis.\n')
    (tex/'jev_probability_methods.tex').write_text(methods,encoding='utf-8')
    results=[r'\subsection{Class-probability diagnostics of Jev-1.14}']
    for ds,cond,title in [('modelnet','views_4','ModelNet with four views'),('s3dis','rgb_aabb','S3DIS with RGB+AABB')]:
        found=next((s for s in stats if s['dataset']==ds and s['condition']==cond and s['vectors_valid']),None)
        if not found: continue
        s=found
        results.append(f'For {title}, complete probability vectors are available for {s["vectors_valid"]} of '
            f'{s["objects_expected"]} evaluated objects. The mean probability assigned to the saved Choice is '
            f'{percentage(s["mean_p_chosen_correct"])} for correct decisions and {percentage(s["mean_p_chosen_incorrect"])} '
            f'for incorrect decisions. The ten-bin ECE is {s["ece_10_bins"]*100:.1f} percentage points. '
            f'The saved Choice is not a maximum-probability class in {s["choice_not_argmax"]} cases; '
            'these decisions are retained as recorded.\n')
    results.append('The probability matrices expose how probability mass is distributed beyond the final decision. '
                   'The selected ModelNet rows match the Jev confusion-matrix selection. '
                   'Complete probability matrices, condition-level diagnostics and coverage are given in '
                   'Appendix~\\ref{app:jev_probabilities}.\n')
    appendix=[r'\clearpage',r'\setcounter{figure}{0}',r'\renewcommand{\thefigure}{B\arabic{figure}}',
              r'\ifdefined\theHfigure\renewcommand{\theHfigure}{appendix.B.\arabic{figure}}\fi',
              r'\setcounter{table}{0}',r'\renewcommand{\thetable}{B\arabic{table}}',
              r'\ifdefined\theHtable\renewcommand{\theHtable}{appendix.B.\arabic{table}}\fi',
              r'\section{Stored Jev Class Probabilities}',r'\label{app:jev_probabilities}',
              'Each condition is evaluated separately on the same object basis as its decision matrix. '
              'Complete vectors are checked against the exact candidate-class set, numeric range and rounding tolerance. '
              'No missing probability mass is added and no vector is renormalized. '
              'Other classes in the focused probability matrix sums the reported probability mass of omitted candidate classes. '
              'It does not denote a new true class.\n',
              r'\begin{table*}[ht]',r'\centering',r'\small',r'\setlength{\tabcolsep}{3pt}',r'\caption{Coverage and descriptive probability diagnostics of Jev-1.14. '
              r'Mean $P(\mathrm{true})$ is probability assigned to the true class. ECE is in percentage points. '
              r'Non-max counts saved Choices whose probability is strictly below the maximum.}',
              r'\label{tab:jev_probability_audit}',r'\begin{tabular}{llrrrrr}',r'\hline',
              r'Dataset & Input & Vectors / objects & Mean $P(\mathrm{true})$ & ECE & Non-max & Sum $\ne 1$ \\',r'\hline']
    for s in stats:
        if not s['vectors_valid']: continue
        condition=(s['condition'].replace('views_','')+(' view' if s['condition']=='views_1' else ' views')) if s['dataset']=='modelnet' else ('RGB' if s['condition']=='rgb' else 'RGB+AABB')
        appendix.append(f'{"ModelNet" if s["dataset"]=="modelnet" else "S3DIS"} & {condition} & '
                        f'{s["vectors_valid"]}/{s["objects_expected"]} & {s["mean_p_true"]*100:.1f}\\% & '
                        f'{s["ece_10_bins"]*100:.1f} & {s["choice_not_argmax"]} & {s["rounded_nonunit_vectors"]} '+r'\\')
    appendix.extend([r'\hline',r'\end{tabular}',r'\end{table*}'])
    validstats=[s for s in stats if s['vectors_valid']]
    appendix.append(f'The sums of accepted vectors range from {min(s["probability_sum_min"] for s in validstats):.2f} '
                    f'to {max(s["probability_sum_max"] for s in validstats):.2f}; deviations from one reflect the saved precision. '
                    'Reported means retain this rounding. Condition-level CSV files also record exclusions, class counts and coverage.\n')
    for figure in figures:
        block=_figure_tex(f'paper_figures/{figure["stem"]}.pdf',figure['caption'],figure['label'])
        if figure['stem'] in ['Jev_ModelNet_probability_focus','Jev_probability_correctness']:
            results.append(block)
        else: appendix.append(block)
    (tex/'jev_probability_results.tex').write_text('\n'.join(results)+'\n',encoding='utf-8')
    (tex/'appendix_jev_probabilities.tex').write_text('\n'.join(appendix)+'\n',encoding='utf-8')
    demo=root/'figures_demo.tex'; source=demo.read_text(encoding='utf-8')
    source=source.replace(r'\input{paper_tex/appendix_modelnet}',r'\input{paper_tex/jev_probability_methods}'+'\n'+
        r'\input{paper_tex/jev_probability_results}'+'\n'+r'\input{paper_tex/appendix_modelnet}'+'\n'+r'\input{paper_tex/appendix_jev_probabilities}')
    demo.write_text(source,encoding='utf-8')


def _render(root: Path, models: list[dict], singles: list[dict], s3dis: list[dict], profile: dict,
            dpi: int, focus_rows: list[dict], metric_rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import PercentFormatter
    figroot = root/'paper_figures'; figroot.mkdir()
    styles={'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,
            'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'savefig.facecolor':'white'}
    with plt.rc_context(styles), PdfPages(figroot/'Abbildungen_Vergleich.pdf') as review, \
         PdfPages(figroot/'ModelNet_Vereinfachte_Matrizen.pdf') as focused_atlas, \
         PdfPages(figroot/'ModelNet_Vollstaendige_Matrizen_Anhang.pdf') as full_atlas:
        def save(fig,stem,*,review_page=False,atlas=None):
            fig.savefig(figroot/f'{stem}.pdf',bbox_inches='tight')
            fig.savefig(figroot/f'{stem}.png',dpi=dpi,bbox_inches='tight')
            if review_page: review.savefig(fig,bbox_inches='tight')
            if atlas is not None: atlas.savefig(fig,bbox_inches='tight')

        # Agreed overview: existing model colours, no duplicate Accuracy/Macro-F1 line page.
        fig,ax=plt.subplots(figsize=(10.4,5.3))
        order={v:i for i,v in enumerate(profile.get('overview_order',[]))}
        for item in sorted(models,key=lambda i:order.get(i['definition']['model'],99)):
            d=item['definition']; views=sorted(item['panels']); values=[panel_metrics(item['panels'][v])['accuracy']*100 for v in views]
            ax.plot(views,values,label=d['title'],color=d['color'],marker='s' if min(views)>1 else 'o',
                    linestyle='--' if min(views)>1 else '-',lw=2.2,ms=5)
        for item in singles:
            d=item['definition']; ax.plot([1],[panel_metrics(item['panel'])['accuracy']*100],marker='s',linestyle='none',color=d['color'],label=d['title'],ms=5)
        ax.set(xticks=[1,2,3,4],xlim=(.85,4.15),ylim=(0,65),xlabel='Number of isometric views',ylabel='Accuracy (%)')
        ax.grid(color='#dddddd',alpha=.65); ax.set_axisbelow(True); ax.legend(ncol=3,fontsize=9,frameon=False,loc='upper left')
        fig.tight_layout(); save(fig,'modelnet_accuracy_by_views',review_page=True); plt.close(fig)

        if s3dis:
            fig,axes=plt.subplots(1,2,figsize=(10.4,5.65),sharey=True)
            for ax,metric,title in zip(axes,['accuracy','macro_f1'],['Accuracy','Macro-F1']):
                for offset,condition,color,name in [(-.18,'rgb','#8fb9dd','RGB'),(.18,'rgb_aabb','#1f5f99','RGB + AABB')]:
                    values=[100*panel_metrics(i['panels'][condition])[metric] for i in s3dis]
                    bars=ax.bar(np.arange(len(s3dis))+offset,values,width=.36,color=color,label=name)
                    ax.bar_label(bars,labels=[f'{v:.1f}' for v in values],padding=3,fontsize=9)
                ax.set_xticks(range(len(s3dis)),[i['definition']['title'] for i in s3dis],rotation=22,ha='right',fontsize=9)
                ax.set(ylim=(0,70),title=title); ax.grid(axis='y',alpha=.3); ax.set_axisbelow(True)
            axes[0].set_ylabel('Performance (%)'); axes[1].legend(frameon=False,fontsize=10)
            fig.suptitle('S3DIS',fontsize=16,y=.98); fig.tight_layout(rect=[0,0,1,.94])
            save(fig,'s3dis_rgb_aabb_performance',review_page=True); plt.close(fig)

        # Four-view class overview: no supports or extra footer, explicit percentages.
        labels=models[0]['labels']
        comparison_order=sorted(models,key=lambda i:order.get(i['definition']['model'],99))
        recalls=np.array([np.diag(panel_matrix(i['panels'][max(i['panels'])],labels)[2]) for i in comparison_order]).T
        fig=plt.figure(figsize=(8.3,9.7)); ax=fig.add_axes([.205,.175,.63,.75])
        im=ax.pcolormesh(np.arange(len(models)+1)-.5,np.arange(len(labels)+1)-.5,recalls,cmap='Blues',vmin=0,vmax=1,edgecolors='white',linewidth=.5)
        ax.set(xlim=(-.5,len(models)-.5),ylim=(len(labels)-.5,-.5))
        ax.set_xticks(range(len(models)),[i['definition']['title'] for i in comparison_order],rotation=35,ha='right',fontsize=10)
        ax.set_yticks(range(len(labels)),[s.replace('_',' ') for s in labels],fontsize=9); ax.tick_params(length=0,pad=4)
        for r,c in np.ndindex(recalls.shape): ax.text(c,r,f'{100*recalls[r,c]:.0f}%',ha='center',va='center',fontsize=8.5,color='white' if recalls[r,c]>=.55 else '#14304b')
        fig.suptitle('ModelNet: correct classifications by class',fontsize=15,y=.975)
        fig.text(.52,.938,'4 views',ha='center',fontsize=11)
        cb=fig.colorbar(im,cax=fig.add_axes([.865,.31,.018,.46]),ticks=np.linspace(0,1,6)); cb.ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0)); cb.set_label('True class',fontsize=10)
        save(fig,'ModelNet_Klassenvergleich_4Views',review_page=True); plt.close(fig)
        pd.DataFrame(recalls,index=labels,columns=[i['definition']['title'] for i in comparison_order]).to_csv(root/'paper_data/ModelNet_Klassenvergleich_4Views.csv')

        for index,item in enumerate(models,1):
            d,f=item['definition'],item['focus']; views=sorted(f['matrices'])
            fig=plt.figure(figsize=(12.2,6.48)); gs=fig.add_gridspec(1,2,left=.105,right=.90,bottom=.315,top=.81,wspace=.09)
            for panel,v in enumerate(views):
                m=f['matrices'][v]; shares=m['shares']; n,k=shares.shape; ax=fig.add_subplot(gs[0,panel])
                im=ax.pcolormesh(np.arange(k+1)-.5,np.arange(n+1)-.5,shares.to_numpy()*100,cmap='Blues',vmin=0,vmax=100,edgecolors='white',linewidth=.5)
                ax.set(xlim=(-.5,k-.5),ylim=(n-.5,-.5),aspect='equal')
                ax.set_xticks(range(k),[s.replace('_',' ') for s in shares.columns],rotation=90,fontsize=9)
                ax.set_yticks(range(n),[s.replace('_',' ') for s in shares.index],fontsize=10); ax.tick_params(length=0,pad=4)
                ax.set_title(f'{v} view'+('s' if v>1 else ''),fontsize=13,pad=10); ax.set_xlabel('Predicted class',fontsize=11,labelpad=8)
                if panel==0: ax.set_ylabel('True class',fontsize=11)
                else: ax.tick_params(axis='y',labelleft=False)
                for r,c in np.ndindex(shares.shape):
                    value=shares.iloc[r,c]
                    if value>0 or r==c: ax.text(c,r,f'{100*value:.0f}%',ha='center',va='center',fontsize=7.8,color='white' if value>=.55 else '#14304b')
                for truth in shares.index:
                    for predicted in shares.columns:
                        focus_rows.append({'model':d['model'],'condition':f'views_{v}','true_class':truth,'predicted_class':predicted,
                            'count':int(m['counts'].loc[truth,predicted]),'support':int(m['support'][truth]),'percentage':100*shares.loc[truth,predicted]})
            fig.suptitle(d['title'],fontsize=19,fontweight='bold',y=.965)
            cb=fig.colorbar(im,cax=fig.add_axes([.93,.39,.013,.32]),ticks=[0,20,40,60,80,100]); cb.ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0)); cb.set_label('True class',fontsize=10)
            note=NOTE if f['approved_gains_still_largest'] else 'Selected classes show changes in recognition and classes that remain difficult to classify.'
            fig.text(.065,.065,note,fontsize=8.8)
            fig.text(.065,.033,f'Complete matrices for all evaluated classes and available view counts: Appendix A, Figure A{index}.',fontsize=8.8)
            save(fig,f'{d["stem"]}_confusion_focus',review_page=True,atlas=focused_atlas); plt.close(fig)

        fig=plt.figure(figsize=(8.27,11.69)); fig.text(.09,.84,'Appendix A',fontsize=25,fontweight='bold',color='#12293e')
        fig.text(.09,.79,'Complete ModelNet confusion matrices',fontsize=17,fontweight='bold')
        fig.text(.09,.71,'Complete matrices for every evaluated true class and reported view configuration.\n\nThe main-text selection illustrates improvements and persistent errors.\nOverall metrics continue to use the full evaluated cohort.',fontsize=11,linespacing=1.6)
        for index,item in enumerate(models,1): fig.text(.09,.53-.033*(index-1),f'A{index}  {item["definition"]["title"]}',fontsize=11)
        review.savefig(fig); plt.close(fig)
        for index,item in enumerate(models,1):
            d=item['definition']; views=sorted(item['panels']); fig=plt.figure(figsize=(17.6 if len(views)==4 else 13.4,4.95))
            grid=fig.add_gridspec(1,len(views),left=.052,right=.94,bottom=.23,top=.86,wspace=.045)
            fig.suptitle(d['title'],fontsize=22,fontweight='bold',y=.98)
            for col,v in enumerate(views):
                raw,support,shares=panel_matrix(item['panels'][v],labels); n=len(labels); ax=fig.add_subplot(grid[0,col])
                im=ax.pcolormesh(np.arange(n+1)-.5,np.arange(n+1)-.5,shares.to_numpy(),cmap='Blues',vmin=0,vmax=1,edgecolors='none',rasterized=False)
                ax.set(xlim=(-.5,n-.5),ylim=(n-.5,-.5),aspect='equal')
                ax.set_xticks(range(n),labels,rotation=90,fontsize=6.2); ax.set_yticks(range(n),labels,fontsize=6.2); ax.tick_params(length=3,width=.65,pad=3)
                for spine in ax.spines.values(): spine.set_visible(True); spine.set_linewidth(.8)
                if col: ax.tick_params(axis='y',labelleft=False)
                else: ax.set_ylabel('True class',fontsize=14)
                ax.set_xlabel('Predicted class',fontsize=14,labelpad=8); ax.set_title(f'{v} view'+('s' if v>1 else ''),fontsize=18,pad=10)
                metric_rows.append({**{k:d[k] for k in ['model','provider','scope']},'dataset':'modelnet','condition':f'views_{v}','view_count':v,**panel_metrics(item['panels'][v]),
                                    'omitted_in_full_matrix':int(support.sum()-raw.reindex(columns=labels,fill_value=0).to_numpy().sum())})
            cb=fig.colorbar(im,cax=fig.add_axes([.958,.31,.009,.52])); cb.set_label('True class',fontsize=13); cb.ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0)); cb.ax.tick_params(labelsize=11)
            # Full files retain the original image, without the rejected historical footer.
            save(fig,f'{d["stem"]}_confusion',atlas=full_atlas)
            fig.text(.02,1.045,f'Appendix A | Figure A{index} | {d["title"]}',fontsize=10,fontweight='bold')
            review.savefig(fig,bbox_inches='tight'); plt.close(fig)


GUIDE = r'''# Auswertungen

Im Reiter Auswertungen gespeicherte Ergebnisse auswählen und Grafiken sowie Daten exportieren. PNGs, PDFs, Zahlen und Herkunftsinformationen stehen im Auswertungspaket bereit. Lokale Daten, Sicherungen und Zugangsdaten gehören nicht in die Veröffentlichung.
'''


def build_publication_export(runs_root: str | Path, output_root: str | Path, *, repeat: int = 1,
                             profile_path: str | Path | None = None, reference_figures: str | Path | None = None,
                             png_dpi: int = 600, historical_csv_root: str | Path | None = None) -> PublicationExport:
    if not 50<=png_dpi<=1200:
        raise ValueError('PNG-Auflösung muss zwischen 50 und 1200 dpi liegen.')
    profile=load_profile(profile_path)
    rows,inventory=load_publication_rows(runs_root)
    models,singles,s3dis,excluded=select_panels(rows,profile,repeat)
    selection_warnings=[]
    for item in models:
        item['labels']=sorted(next(iter(item['panels'].values())).true_label.unique())
        item['focus']=focus_matrices(item['panels'],item['definition'],item['labels'])
        if not item['focus']['approved_gains_still_largest']:
            selection_warnings.append(f"{item['definition']['title']}: Die festgelegten Verbesserungsklassen sind in diesen Daten nicht mehr alle die größten Verbesserungen.")
    output=Path(output_root).expanduser().resolve()
    runs=Path(runs_root).expanduser().resolve()
    if output==runs or runs in output.parents:
        raise ValueError('Der Grafikexport darf nicht innerhalb des Ergebnislauf-Ordners liegen.')
    output.parent.mkdir(parents=True,exist_ok=True)
    # Stage an entire generation before replacing files so failed renders keep the last export usable.
    with tempfile.TemporaryDirectory(prefix='paper_figures_',dir=output.parent) as temporary:
        stage=Path(temporary); (stage/'paper_data').mkdir()
        focus_rows,metric_rows=[],[]
        _render(stage,models,singles,s3dis,profile,png_dpi,focus_rows,metric_rows)
        for item in s3dis:
            for condition,panel in item['panels'].items():
                metric_rows.append({'model':item['definition']['model'],'provider':item['definition']['provider'],'scope':item['definition']['scope'],
                                    'dataset':'s3dis','condition':condition,'view_count':4,**panel_metrics(panel)})
        for item in singles:
            metric_rows.append({'model':item['definition']['model'],'provider':item['definition']['provider'],'scope':item['definition']['scope'],
                                'dataset':'modelnet','condition':'views_1','view_count':1,**panel_metrics(item['panel'])})
        pd.DataFrame(focus_rows).to_csv(stage/'paper_data/ModelNet_Vereinfachte_Matrizen.csv',index=False)
        pd.DataFrame(metric_rows).to_csv(stage/'paper_data/figure_metrics.csv',index=False)
        pd.DataFrame(excluded).to_csv(stage/'paper_data/excluded_panels.csv',index=False)
        _write_latex(stage,models,profile,bool(s3dis))
        _write_metric_tables(stage,models,singles,s3dis)
        s3dis_confusions=write_s3dis_confusions(stage,s3dis,png_dpi)
        probability_panels,probability_summary=collect_probability_panels(Path(runs_root),models,s3dis)
        probability_figures=render_probability_exports(stage,probability_panels,probability_summary,png_dpi)
        if probability_figures:
            _write_probability_latex(stage,probability_figures,probability_summary)
        s3dis_paired=write_s3dis_paired_analysis(stage,s3dis,png_dpi,appendix_letter='C' if probability_figures else 'B')
        pilot_info=None
        if historical_csv_root:
            pilot=save_pilot_figure(historical_csv_root,Path(__file__).resolve().parents[1]/'Evaluator_stablediffusion.py',
                                   stage,dpi=png_dpi,package_layout=True)
            pilot_info=pilot['manifest']
            demo=stage/'figures_demo.tex'
            demo.write_text(demo.read_text(encoding='utf-8').replace(r'\begin{document}',
                r'\begin{document}'+'\n'+r'\section{Historical Pilot}\input{paper_tex/pilot_confusion}'+'\n',1),encoding='utf-8')
        (stage/'PAPER_FIGURES_GUIDE.md').write_text(GUIDE,encoding='utf-8')
        (stage/'paper_figure_profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
        references=[]
        if reference_figures:
            assets=Path(reference_figures).expanduser().resolve()
            if not assets.is_dir(): raise ValueError('Der Ordner mit Originalabbildungen existiert nicht.')
            generated={p.name for p in (stage/'paper_figures').iterdir()}
            for path in sorted(assets.iterdir()):
                if path.is_file() and path.suffix.lower() in ['.pdf','.png','.jpg','.jpeg','.eps'] and path.name not in generated:
                    target=stage/'paper_figures/reference'/path.name; target.parent.mkdir(exist_ok=True)
                    shutil.copy2(path,target)
                    references.append({'file':str(target.relative_to(stage)).replace('\\','/'),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest={'format_version':1,'profile_version':profile['version'],'created_utc':datetime.now(timezone.utc).isoformat(),
            'generator_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ['publication_export.py','jev_probability_analysis.py','s3dis_publication.py','s3dis_paired_analysis.py','pilot_publication.py','paper_figure_profile.json']},
            'repeat':repeat,'png_dpi':png_dpi,'runs':inventory,'models':[i['definition']['model'] for i in models],
            'appendix_order':[{'model':i['definition']['model'],'figure':f'A{n}'} for n,i in enumerate(models,1)],
            'class_overview_figure':f'A{len(models)+1}','s3dis_models':[i['definition']['model'] for i in s3dis],
            's3dis_confusions':s3dis_confusions,
            's3dis_paired_analysis':s3dis_paired,
            'pilot_figure7':pilot_info,
            'selection_warnings':selection_warnings,'excluded_panels':excluded,'reference_assets':references,
            'jev_probabilities':{'conditions':probability_summary,'figures':probability_figures,
                'normalization':'none; complete finite Choice vectors with total within 0.01 of 1',
                'confidence':'probability assigned to saved Choice; separate API confidence not used',
                'ece':'ten fixed-width bins; sample-weighted absolute accuracy minus mean P(saved Choice)',
                'historical_noul':'independent yes/no scores excluded'},
            'invalid_responses':'retained in all matrix denominators and counted as incorrect for accuracy',
            'macro_f1':'valid responses only; same union of observed true and predicted labels as existing evaluation',
            'selections':[{'model':i['definition']['model'],'true_classes':i['focus']['rows'],
                'extra_prediction_classes':i['focus']['extras'],'approved_gains_still_largest':i['focus']['approved_gains_still_largest']} for i in models]}
        manifest['files']={str(p.relative_to(stage)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.rglob('*')) if p.is_file()}
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        output.mkdir(parents=True,exist_ok=True)
        # Remove only previously generated files tracked by this export, never arbitrary user files.
        previous=output/'manifest.json'
        if previous.is_file():
            previous_files=json.loads(previous.read_text(encoding='utf-8')).get('files',{})
            for relative in set(previous_files)-set(manifest['files']):
                old=(output/relative).resolve()
                if output in old.parents and old.is_file(): old.unlink()
        for source in stage.rglob('*'):
            if source.is_file():
                target=output/source.relative_to(stage); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
    zip_path=output/'MMFM_Paper_LaTeX.zip'
    members=[*manifest['files'],'manifest.json']
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(members): archive.write(output/relative,arcname=relative)
    return PublicationExport(output,zip_path,manifest)


def main() -> None:
    parser=argparse.ArgumentParser(description='Export saved paper figures and ready-to-use LaTeX inputs without API calls.')
    parser.add_argument('--runs',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--repeat',type=int,default=1); parser.add_argument('--profile')
    parser.add_argument('--reference-figures'); parser.add_argument('--png-dpi',type=int,default=600)
    parser.add_argument('--historical-csv',help='Optional original pilot CSV archive, including Stable Diffusion captions.')
    args=parser.parse_args()
    result=build_publication_export(args.runs,args.output,repeat=args.repeat,profile_path=args.profile,
                                    reference_figures=args.reference_figures,png_dpi=args.png_dpi,historical_csv_root=args.historical_csv)
    print(json.dumps({'output':str(result.output_dir),'zip':str(result.zip_path),
                      'models':result.manifest['models'],'warnings':result.manifest['selection_warnings']},indent=2))


if __name__=='__main__': main()
