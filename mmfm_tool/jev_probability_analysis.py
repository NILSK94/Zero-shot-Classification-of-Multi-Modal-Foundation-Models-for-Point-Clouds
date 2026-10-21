"""Describe stored Jev Choice probabilities without inventing probability mass.

The API's separate confidence field is retained for audit only. Historical Noul
yes/no scores are independent scores and are never treated as Choice vectors.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROUNDING_TOLERANCE = 0.010000001


def parse_choice(raw: str, labels: list[str], prediction: str) -> dict:
    try:
        answer = json.loads(raw)['answers']['classification']
    except (ValueError,TypeError,KeyError) as exc:
        raise ValueError('missing Choice response') from exc
    if answer.get('type') != 'choice':
        raise ValueError('not a Choice probability vector (independent Noul scores excluded)')
    probabilities = answer.get('probabilities')
    if not isinstance(probabilities,dict) or set(probabilities)!=set(labels):
        raise ValueError('incomplete or different class set')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in probabilities.values()):
        raise ValueError('non-numeric probability')
    values = np.array([probabilities[k] for k in labels],dtype=float)
    if not np.isfinite(values).all() or (values<0).any() or (values>1).any():
        raise ValueError('non-finite or out-of-range probability')
    total = float(values.sum())
    if abs(total-1)>ROUNDING_TOLERANCE:
        raise ValueError('probability sum outside documented rounding tolerance')
    if answer.get('choice')!=prediction or prediction not in labels:
        raise ValueError('Choice and saved prediction disagree')
    confidence=answer.get('confidence')
    if isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not np.isfinite(confidence):
        confidence=np.nan
    chosen=float(probabilities[prediction])
    return {'values':values,'probability_sum':total,'p_chosen':chosen,
            'p_max':float(values.max()),'choice_is_argmax':bool(np.isclose(chosen,values.max(),atol=1e-12,rtol=0)),
            'api_confidence':confidence}


def calibration_bins(frame: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Observed correctness of the saved Choice vs its own probability, not p_max."""
    if bins<1: raise ValueError('At least one bin is required.')
    indices=np.minimum(np.floor(np.round(frame.p_chosen.to_numpy()*bins,12)).astype(int),bins-1)
    rows=[]
    for index in range(bins):
        selected=frame.iloc[np.flatnonzero(indices==index)]
        rows.append({'lower':index/bins,'upper':(index+1)/bins,'objects':len(selected),
                     'mean_probability':float(selected.p_chosen.mean()) if len(selected) else np.nan,
                     'accuracy':float(selected.is_correct.mean()) if len(selected) else np.nan})
    return pd.DataFrame(rows)


def probability_matrix(frame: pd.DataFrame, labels: list[str], *, rows: list[str] | None = None,
                       columns: list[str] | None = None) -> pd.DataFrame:
    matrix=frame.groupby('true_label')[[f'p::{c}' for c in labels]].mean()
    matrix.columns=labels
    matrix=matrix.reindex(rows or labels)
    if columns is not None:
        shown=[c for c in columns if c in labels]
        remaining=[c for c in labels if c not in shown]
        matrix=matrix[shown].assign(**{'Other classes':matrix[remaining].sum(axis=1,min_count=1) if remaining else 0.0})
    return matrix


def collect_probability_panels(runs_root: Path, models: list[dict], s3dis: list[dict]) -> tuple[list[dict],list[dict]]:
    results,summary=[],[]
    cache={}
    for item in models+s3dis:
        definition=item['definition']
        if definition['provider']!='jev' or definition['model']!='typesafe/jev-1.14': continue
        dataset='modelnet' if 'focus' in item else 's3dis'
        labels=sorted(next(iter(item['panels'].values())).true_label.unique())
        collected={}
        for condition,panel in item['panels'].items():
            valid,excluded=[],[]
            for _,record in panel.iterrows():
                run=record.source_run
                if run not in cache:
                    source=Path(runs_root)/run/'results.csv'
                    header=pd.read_csv(source,nrows=0).columns
                    cache[run]=pd.read_csv(source,usecols=['raw_response']).fillna('') if 'raw_response' in header else None
                raw=cache[run]
                try:
                    if raw is None: raise ValueError('no saved response')
                    if not record['_valid']: raise ValueError('invalid saved classification')
                    parsed=parse_choice(raw.iloc[int(record['_source_row'])].raw_response,labels,record.pred_label)
                    values=parsed.pop('values')
                    valid.append({'sample_id':record.sample_id,'true_label':record.true_label,'pred_label':record.pred_label,
                                  'is_correct':bool(record.is_correct),**parsed,**{f'p::{c}':v for c,v in zip(labels,values)}})
                except ValueError as exc:
                    excluded.append({'sample_id':record.sample_id,'reason':str(exc)})
            frame=pd.DataFrame(valid)
            n=len(frame)
            cond=f'views_{condition}' if dataset=='modelnet' else str(condition)
            stats={'dataset':dataset,'condition':cond,'objects_expected':len(panel),'vectors_valid':n,
                   'coverage':n/len(panel),'excluded':len(excluded),'classes':len(labels)}
            if n:
                bins=calibration_bins(frame); occupied=bins[bins.objects.gt(0)]
                stats.update({'accuracy_valid_vectors':float(frame.is_correct.mean()),
                    'mean_p_true':float(np.mean([r[f'p::{r.true_label}'] for _,r in frame.iterrows()])),
                    'mean_p_chosen':float(frame.p_chosen.mean()),
                    'mean_p_chosen_correct':float(frame.loc[frame.is_correct,'p_chosen'].mean()) if frame.is_correct.any() else None,
                    'mean_p_chosen_incorrect':float(frame.loc[~frame.is_correct,'p_chosen'].mean()) if (~frame.is_correct).any() else None,
                    'probability_sum_min':float(frame.probability_sum.min()),'probability_sum_max':float(frame.probability_sum.max()),
                    'rounded_nonunit_vectors':int((abs(frame.probability_sum-1)>1e-8).sum()),
                    'choice_not_argmax':int((~frame.choice_is_argmax).sum()),
                    'ece_10_bins':float((occupied.objects*(occupied.accuracy-occupied.mean_probability).abs()).sum()/n)})
                coverage=panel.groupby('true_label').size().rename('expected').to_frame()
                coverage['accepted']=frame.groupby('true_label').size().reindex(coverage.index,fill_value=0)
                collected[condition]={'frame':frame,'bins':bins,'stats':stats,'excluded':excluded,'class_coverage':coverage}
            summary.append(stats)
        if collected:
            results.append({'dataset':dataset,'labels':labels,'panels':collected,'source_item':item})
    return results,summary


def render_probability_exports(root: Path, panels: list[dict], stats: list[dict], dpi: int) -> list[dict]:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import PercentFormatter
    figures=[]
    data=root/'paper_data'; plots=root/'paper_figures'
    pd.DataFrame(stats).to_csv(data/'Jev_probability_summary.csv',index=False)
    if not panels: return figures
    def save(fig,stem,caption,label):
        fig.savefig(plots/f'{stem}.pdf',bbox_inches='tight')
        fig.savefig(plots/f'{stem}.png',dpi=dpi,bbox_inches='tight')
        atlas.savefig(fig,bbox_inches='tight'); plt.close(fig)
        figures.append({'stem':stem,'caption':caption,'label':label})
    def matrix_figure(item,conditions,focus=False):
        labels=item['labels']; source=item['source_item']; subset=source.get('focus',{})
        rowlabels=subset.get('rows') if focus else labels
        columns=[c for c in subset.get('columns',[]) if c in labels] if focus else None
        height=6.6 if focus else (8 if len(labels)>20 else 6.5)
        fig,axes=plt.subplots(1,len(conditions),figsize=(13.6,height),squeeze=False)
        fig.suptitle(f'Jev-1.14 | {"ModelNet" if item["dataset"]=="modelnet" else "S3DIS"}',fontsize=19,y=1.01)
        for ax,condition in zip(axes[0],conditions):
            part=item['panels'][condition]; frame=part['frame']
            matrix=probability_matrix(frame,labels,rows=rowlabels,columns=columns)
            cmap=plt.get_cmap('Blues').copy(); cmap.set_bad('#dedede')
            im=ax.imshow(matrix.to_numpy(),vmin=0,vmax=1,cmap=cmap,aspect='auto')
            ax.set_xticks(range(len(matrix.columns)),[s.replace('_',' ') for s in matrix.columns],rotation=90,fontsize=9 if focus else 7)
            ax.set_yticks(range(len(matrix.index)),[s.replace('_',' ') for s in matrix.index],fontsize=10 if focus else 8)
            title=(f'{condition} view'+('s' if condition!=1 else '')) if item['dataset']=='modelnet' else ('RGB' if condition=='rgb' else 'RGB + AABB')
            ax.set_title(title,fontsize=14,pad=12)
            ax.set_xlabel('Class receiving probability',labelpad=9)
            ax.set_ylabel('True class' if ax==axes[0,0] else '')
            ax.set_xticks(np.arange(-.5,len(matrix.columns)),minor=True)
            ax.set_yticks(np.arange(-.5,len(matrix.index)),minor=True)
            ax.grid(which='minor',color='white',linewidth=.6); ax.tick_params(which='minor',length=0)
            if focus or len(labels)<=13:
                for y,truth in enumerate(matrix.index):
                    for x,destination in enumerate(matrix.columns):
                        value=matrix.iloc[y,x]
                        if np.isfinite(value) and (value>=.005 or truth==destination):
                            ax.text(x,y,f'{value*100:.0f}%',ha='center',va='center',fontsize=8,
                                    color='white' if value>.5 else '#29475e')
        fig.subplots_adjust(left=.10,right=.91,bottom=.26,top=.9,wspace=.34)
        color=fig.add_axes([.935,.29,.012,.57]); fig.colorbar(im,cax=color,format=PercentFormatter(1)).set_label('Mean reported class probability' if focus else 'True class')
        return fig
    with plt.rc_context({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42}), PdfPages(plots/'Jev_Wahrscheinlichkeiten.pdf') as atlas:
        allrows=[]; allbins=[]
        for item in panels:
            ds=item['dataset']; conditions=[min(item['panels']),max(item['panels'])] if ds=='modelnet' else ['rgb','rgb_aabb']
            conditions=list(dict.fromkeys(c for c in conditions if c in item['panels']))
            for condition,part in item['panels'].items():
                frame=part['frame'].copy(); frame.insert(0,'condition',f'views_{condition}' if ds=='modelnet' else condition); frame.insert(0,'dataset',ds)
                allrows.append(frame)
                bins=part['bins'].copy(); bins.insert(0,'condition',frame.condition.iloc[0]); bins.insert(0,'dataset',ds); allbins.append(bins)
                probability_matrix(part['frame'],item['labels']).to_csv(data/f'Jev_{ds}_{condition}_mean_probability.csv',index_label='true_class')
                pd.DataFrame(part['excluded'],columns=['sample_id','reason']).to_csv(data/f'Jev_{ds}_{condition}_probability_exclusions.csv',index=False)
                part['class_coverage'].to_csv(data/f'Jev_{ds}_{condition}_probability_class_coverage.csv',index_label='true_class')
            if ds=='modelnet':
                save(matrix_figure(item,conditions,True),'Jev_ModelNet_probability_focus',
                    'Mean reported class probabilities of Jev-1.14 for the same selected ModelNet classes as the decision matrices, '
                    f'with {" and ".join(str(c) for c in conditions)} input views. Rows average the saved probability vectors within each true class; these are not confusion counts. '
                    'Other classes sums probability assigned to omitted candidate classes. Values are not renormalized. '
                    'Complete probability matrices and the rounding audit are provided in Appendix~\\ref{app:jev_probabilities}.',
                    'fig:jev_probability_focus')
            save(matrix_figure(item,conditions),f'Jev_{ds}_probability_full',
                f'Mean reported class probabilities of Jev-1.14 on {"ModelNet" if ds=="modelnet" else "S3DIS"}. '
                'The common scale averages probability mass within each true class, rather than counting final decisions. '
                'All candidate classes are shown. Saved vectors are used without renormalization; rounded totals can equal 0.99.',
                f'fig:jev_{ds}_probability_full')
        pd.concat(allrows,ignore_index=True).to_csv(data/'Jev_probability_vectors.csv',index=False)
        pd.concat(allbins,ignore_index=True).to_csv(data/'Jev_probability_bins.csv',index=False)
        fig,axes=plt.subplots(2,len(panels),figsize=(11.8,8),squeeze=False)
        fig.suptitle('Jev-1.14: reported probability and observed correctness',fontsize=17,y=.99)
        correctness_conditions=[]
        for col,item in enumerate(panels):
            condition=max(item['panels']) if item['dataset']=='modelnet' else ('rgb_aabb' if 'rgb_aabb' in item['panels'] else next(iter(item['panels'])))
            part=item['panels'][condition]; frame,bins=part['frame'],part['bins']; occupied=bins[bins.objects.gt(0)]
            ax=axes[0,col]; ax.plot([0,1],[0,1],color='#999999',ls='--',lw=1)
            ax.plot(occupied.mean_probability,occupied.accuracy,'o-',color='#1f5f99')
            for _,r in occupied.iterrows(): ax.annotate(f'n={int(r.objects)}',(r.mean_probability,r.accuracy),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7)
            title=(f'ModelNet | {condition} view'+('s' if condition!=1 else '')) if item['dataset']=='modelnet' else ('S3DIS | '+('RGB + AABB' if condition=='rgb_aabb' else 'RGB'))
            correctness_conditions.append(title.replace(' | ',' with '))
            ax.set(xlim=(0,1),ylim=(0,1.08),xlabel='Mean reported P(chosen class)',ylabel='Observed accuracy',title=title)
            ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.15)
            ax=axes[1,col]; centers=(np.arange(10)+.5)/10
            for offset,correct,color,label in [(-.021,True,'#1f5f99','Correct decisions'),(.021,False,'#8fb9dd','Incorrect decisions')]:
                counts=calibration_bins(frame.loc[frame.is_correct.eq(correct)]).objects
                ax.bar(centers+offset,counts,width=.042,color=color,label=label)
            ax.set(xlabel='Reported P(chosen class)',ylabel='Number of objects',xlim=(0,1)); ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.legend(fontsize=9)
            ax.spines[['top','right']].set_visible(False)
        fig.tight_layout(rect=[0,0,1,.95],h_pad=3)
        save(fig,'Jev_probability_correctness',
            f'Reported probability assigned to the saved Choice versus observed correctness: {"; ".join(correctness_conditions)}. '
            'The upper panels use ten fixed probability bins (the last includes 1); labels show object counts. '
            'The lower panels retain the counts of correct and incorrect decisions. '
            'The separate API confidence field is not used. These descriptive, single-run plots do not establish calibration.',
            'fig:jev_probability_correctness')
    return figures
