"""Faithful 3x4 reproduction of the Phase-1 figure with explicit historical counting.

Stable Diffusion is a replay of the historical ground-truth-dependent caption
matching rule, never an inferred independent multiclass classifier prediction.
"""
from __future__ import annotations
import ast
import csv
from difflib import get_close_matches
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from .historical_csv import load_historical_csvs

STEM='Pilot_Figure7_confusion_matrices'


def replay_caption_evaluation(path: Path, evaluator: Path) -> tuple[pd.DataFrame,dict]:
    tree=ast.parse(evaluator.read_text(encoding='utf-8'))
    definitions=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYNONYMS' for t in n.targets)]
    if len(definitions)!=1:raise ValueError('Historical synonym dictionary is missing or ambiguous.')
    synonyms=ast.literal_eval(definitions[0].value)
    if not isinstance(synonyms,dict) or not all(isinstance(k,str) and isinstance(v,list) and all(isinstance(s,str) for s in v) for k,v in synonyms.items()):
        raise ValueError('Invalid historical synonym dictionary.')
    records=[]; repaired=0
    with path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.reader(stream)
        if next(reader,None)!=['filename','prompt']:raise ValueError('Unexpected Stable Diffusion source header.')
        for line,row in enumerate(reader,2):
            if len(row)==1:
                row=next(csv.reader([row[0]])); repaired+=1
            if len(row)!=2 or not row[0] or not row[1]:raise ValueError(f'Unusable caption record at line {line}.')
            filename,prompt=row
            truth=filename.split('_')[0].lower()
            candidates=synonyms.get(truth,[truth])
            words=prompt.lower().replace(',','').replace('.','').split()
            matched=any(get_close_matches(word,candidates,n=1,cutoff=.8) for word in words)
            records.append({'sample_id':filename,'true_label':truth,'pred_label':truth if matched else 'unknown',
                'source_row':line,'caption_match':bool(matched)})
    result=pd.DataFrame(records)
    if result.empty or result.sample_id.duplicated().any():raise ValueError('Empty or duplicate caption objects.')
    return result,{'method':'historical ground-truth-dependent synonym matching; unsuccessful matches = unknown',
                   'cutoff':.8,'repaired_outer_quoting_rows':repaired,
                   'evaluator_sha256':hashlib.sha256(evaluator.read_bytes()).hexdigest()}


def pilot_panels(archive_root: Path, evaluator: Path) -> tuple[list[dict],dict]:
    data,_=load_historical_csvs(archive_root,original_rows=True)
    if data.empty:raise ValueError('No historical pilot records found.')
    panels=[]
    def add(row,col,dataset,model,scope,condition,title):
        panel=data[data.dataset.eq(dataset)&data.model.eq(model)&data.protocol_scope.eq(scope)&data.condition.eq(condition)].copy()
        if panel.empty:raise ValueError(f'Missing pilot panel: {title}, {model}, {condition}.')
        if panel.status.ne('ok').any():raise ValueError('Invalid archived responses require an explicit display policy.')
        panels.append({'row':row,'column':col,'dataset':dataset,'model':model,'condition':condition,
            'title':title,'data':panel,'source_files':sorted(panel.source_run.unique()),
            'counting':'original CSV records' if dataset=='modelnet' else 'object decisions'})
    for row,model in [(1,'gemini-1.5-pro'),(2,'gpt-4o')]:
        for view in range(1,5):
            add(row,view,'modelnet',model,'historical_modelnet_pilot',f'views_{view}',f'{view} view'+('s' if view>1 else ''))
    stable_paths=list(archive_root.glob('Stable-diffusion*big-test.csv'))
    if len(stable_paths)!=1:raise ValueError('Stable Diffusion caption archive is missing or ambiguous.')
    stable,replay=replay_caption_evaluation(stable_paths[0],evaluator)
    panels.append({'row':3,'column':1,'dataset':'modelnet','model':'stable-diffusion','condition':'views_1',
        'title':'Stable Diffusion\nCaption matching','data':stable,'source_files':[stable_paths[0].name],
        'counting':'historical caption-match outcomes'})
    add(3,2,'modelnet','gemini-1.5-pro','historical_modelnet_full','views_1','Gemini-1.5-pro\nModelNet larger test')
    add(3,3,'blk2go','gemini-1.5-pro','historical_blk2go_pilot','rgb','Gemini-1.5-pro\nBLK2GO RGB')
    add(3,4,'blk2go','gemini-1.5-pro','historical_blk2go_pilot','rgb_aabb','Gemini-1.5-pro\nBLK2GO RGB + AABB')
    for panel in panels:
        frame=panel['data']; labels=sorted(set(frame.true_label)|set(frame.pred_label))
        if panel['row']<3:
            grouped=frame.groupby('sample_id')
            if not grouped.size().eq(panel['column']).all() or not grouped.pred_label.nunique().eq(1).all():
                raise ValueError('Pilot archive does not contain one repeated object decision per supplied view.')
        counts=pd.crosstab(frame.true_label,frame.pred_label).reindex(index=labels,columns=labels,fill_value=0)
        if int(counts.to_numpy().sum())!=len(frame):raise ValueError('Confusion counts omit historical records.')
        if int(np.trace(counts.to_numpy()))!=int(frame.true_label.eq(frame.pred_label).sum()):raise ValueError('Confusion diagonal disagrees with the archived outcomes.')
        panel['counts']=counts
    return panels,replay


def save_pilot_figure(archive_root: str|Path, evaluator: str|Path, output_root: str|Path, *, dpi: int=600, package_layout: bool=False) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    archive_root,evaluator,root=Path(archive_root),Path(evaluator),Path(output_root)
    panels,replay=pilot_panels(archive_root,evaluator)
    root.mkdir(parents=True,exist_ok=True)
    image_root=root/'paper_figures' if package_layout else root
    data_root=root/'paper_data' if package_layout else root
    tex_root=root/'paper_tex' if package_layout else root
    for folder in [image_root,data_root,tex_root]:folder.mkdir(parents=True,exist_ok=True)
    cells=[]; observations=[]; metrics=[]
    for panel in panels:
        frame=panel['data']; counts=panel['counts']
        for truth in counts.index:
            for predicted in counts.columns:
                cells.append({'row':panel['row'],'column':panel['column'],'model':panel['model'],
                    'condition':panel['condition'],'true_label':truth,'pred_label':predicted,
                    'count':int(counts.loc[truth,predicted]),'true_class_records':int(counts.loc[truth].sum()),
                    'counting':panel['counting']})
        columns=[c for c in ['sample_id','true_label','pred_label','historical_source_rows','source_row'] if c in frame]
        records=frame[columns].copy(); records.insert(0,'panel',f'{panel["row"]}-{panel["column"]}')
        records.insert(1,'counting',panel['counting']); observations.append(records)
        metrics.append({'row':panel['row'],'column':panel['column'],'model':panel['model'],'dataset':panel['dataset'],
            'condition':panel['condition'],'objects':int(frame.sample_id.nunique()),'records':len(frame),
            'correct_records':int(np.trace(counts.to_numpy())),'accuracy':float(frame.true_label.eq(frame.pred_label).mean()),
            'macro_f1':float(f1_score(frame.true_label,frame.pred_label,average='macro',zero_division=0)),
            'counting':panel['counting'],'color_max_count':int(counts.to_numpy().max()),'source_files':';'.join(panel['source_files'])})
    pd.DataFrame(cells).to_csv(data_root/f'{STEM}_counts.csv',index=False)
    pd.concat(observations,ignore_index=True).to_csv(data_root/f'{STEM}_records.csv',index=False)
    pd.DataFrame(metrics).to_csv(data_root/f'{STEM}_metrics.csv',index=False)
    with plt.rc_context({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'}):
        fig,axes=plt.subplots(3,4,figsize=(20,15.3))
        fig.subplots_adjust(left=.062,right=.985,bottom=.075,top=.95,wspace=.27,hspace=.62)
        for panel in panels:
            row,col=panel['row']-1,panel['column']-1; ax=axes[row,col]; counts=panel['counts']; labels=list(counts.index); n=len(labels)
            vmax=max(1,int(counts.to_numpy().max()))
            ax.pcolormesh(np.arange(n+1)-.5,np.arange(n+1)-.5,counts.to_numpy(),cmap='Blues',vmin=0,vmax=vmax,edgecolors='none',rasterized=False)
            ax.set(xlim=(-.5,n-.5),ylim=(n-.5,-.5),aspect='equal')
            ax.set_xticks(range(n),labels,rotation=90,fontsize=6.1 if n>20 else 8)
            # Keep independent panel taxonomies visible, including archival aliases.
            ax.set_yticks(range(n),labels,fontsize=6.1 if n>20 else 8)
            ax.tick_params(length=2.2,width=.55,pad=2)
            ax.set_xlabel('Predicted class',fontsize=9,labelpad=6)
            if col==0 or row==2:ax.set_ylabel('True class',fontsize=9,labelpad=5)
            ax.set_title(panel['title'],fontsize=13 if row<2 else 11,pad=12)
            for y,x in np.argwhere(counts.to_numpy()>0):
                value=counts.iloc[y,x]
                ax.text(x,y,str(value),ha='center',va='center',fontsize=4.6 if n>20 else 8,
                        color='white' if value/vmax>=.55 else '#14304b')
        y1=(axes[0,0].get_position().y0+axes[0,0].get_position().y1)/2
        y2=(axes[1,0].get_position().y0+axes[1,0].get_position().y1)/2
        fig.text(.012,y1,'Gemini-1.5-pro',rotation=90,ha='center',va='center',fontsize=19)
        fig.text(.012,y2,'GPT-4o',rotation=90,ha='center',va='center',fontsize=19)
        divider=(axes[1,0].get_position().y0+axes[2,0].get_position().y1)/2-.025
        fig.add_artist(Line2D([.04,.985],[divider,divider],transform=fig.transFigure,color='#36414d',linewidth=1.0))
        fig.savefig(image_root/f'{STEM}.pdf',bbox_inches='tight')
        fig.savefig(image_root/f'{STEM}.png',dpi=dpi,bbox_inches='tight'); plt.close(fig)
    caption=('Confusion matrices of the historical ModelNet and BLK2GO experiments. '
        'First row: Gemini-1.5-pro with one to four isometric views. Second row: GPT-4o with one to four isometric views. '
        'Third row, from left to right: Stable Diffusion caption matching on the larger ModelNet test set; '
        'Gemini-1.5-pro on the larger ModelNet test set; and Gemini-1.5-pro on BLK2GO with RGB input and RGB+AABB input. '
        'Cell values reproduce the archived record counts; zero entries are omitted and each panel uses its own count scale. '
        'The multi-view ModelNet pilot CSVs repeat object predictions for the supplied views; these records are not additional independent objects. '
        'The Stable Diffusion panel reproduces the historical synonym-matching evaluation, assigning a successful match to the true class and an unsuccessful match to unknown; it is not an independent multiclass prediction.')
    (tex_root/f'{STEM}_caption.tex').write_text('\\caption{'+caption+'}\n\\label{fig:confusion}\n',encoding='utf-8')
    snippet=('\\begin{figure}[H]\n\\centering\n'
        f'\\includegraphics[width=\\textwidth,height=0.82\\textheight,keepaspectratio]{{paper_figures/{STEM}.png}}\n'
        '\\caption{'+caption+'}\n\\label{fig:confusion}\n\\end{figure}\n')
    (tex_root/'pilot_confusion.tex').write_text(snippet,encoding='utf-8')
    source_files=sorted({name for p in panels for name in p['source_files']})
    manifest={'format_version':1,'layout':'3 rows x 4 columns','png_dpi':dpi,'caption':caption,'stable_diffusion':replay,
        'historical_truth':'Original notebook: filename before first underscore; archival aliases are not silently merged.',
        'counting':'ModelNet: original CSV records. BLK2GO: one decision per object.',
        'panels':metrics,'sources':{name:hashlib.sha256((archive_root/name).read_bytes()).hexdigest() for name in source_files}}
    generated=[data_root/f'{STEM}_{suffix}.csv' for suffix in ['counts','records','metrics']]+[
        image_root/f'{STEM}.png',image_root/f'{STEM}.pdf',tex_root/f'{STEM}_caption.tex',tex_root/'pilot_confusion.tex']
    manifest['files']={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in generated}
    (data_root/f'{STEM}_provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return {'root':root,'manifest':manifest,'png':image_root/f'{STEM}.png','pdf':image_root/f'{STEM}.pdf'}
