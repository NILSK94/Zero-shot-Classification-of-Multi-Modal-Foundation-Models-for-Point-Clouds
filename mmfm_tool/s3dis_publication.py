"""Hard-decision S3DIS matrices from saved object predictions, never probabilities."""
from pathlib import Path
import numpy as np
import pandas as pd

STEM = 's3dis_confusion_rgb_aabb_4models'


def validate_panels(items: list[dict]) -> list[str]:
    reference=None
    for item in items:
        if set(item['panels'])!={'rgb','rgb_aabb'}:
            raise ValueError('S3DIS requires both RGB and RGB+AABB.')
        for condition,panel in item['panels'].items():
            if panel.empty or panel.sample_id.duplicated().any():
                raise ValueError('S3DIS contains empty panels or duplicate object IDs.')
            truth=panel.set_index('sample_id').true_label.to_dict()
            if reference is None: reference=truth
            if truth!=reference:
                raise ValueError('S3DIS panels must use exactly the same objects and ground truth.')
    return sorted(set(reference.values())) if reference else []


def decision_matrix(panel: pd.DataFrame, labels: list[str]):
    support=panel.groupby('true_label').size().reindex(labels,fill_value=0)
    if support.eq(0).any(): raise ValueError('True classes with no evaluated objects.')
    valid=panel['_valid'].astype(bool)
    # Reconstruct from saved hard labels; a stored probability argmax is never substituted.
    prediction=panel.pred_label.where(valid,'[No valid response]')
    columns=labels+sorted(set(prediction)-set(labels))
    counts=pd.crosstab(panel.true_label,prediction).reindex(index=labels,columns=columns,fill_value=0)
    fractions=counts.div(support,axis=0)
    correct=int((valid & panel.true_label.eq(panel.pred_label)).sum())
    if int(np.trace(counts[labels].to_numpy()))!=correct:
        raise ValueError('Confusion diagonal and object accuracy disagree.')
    if not np.array_equal(counts.sum(axis=1),support):
        raise ValueError('Confusion counts do not preserve every evaluated object.')
    return counts,support,fractions[labels]


def write_s3dis_confusions(root: Path, items: list[dict], dpi: int = 600) -> dict:
    """Write combined, individual, hard-label data and portable LaTeX artifacts."""
    labels=validate_panels(items)
    if not labels: return {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plots=root/'paper_figures'; data=root/'paper_data'; tex=root/'paper_tex'
    for folder in [plots,data,tex]:folder.mkdir(parents=True,exist_ok=True)
    audit,cells,decisions=[],[],[]
    prepared={}
    for item in items:
        definition=item['definition']; model=definition['model']
        prepared[model]={}
        for condition,panel in item['panels'].items():
            counts,support,shares=decision_matrix(panel,labels)
            prepared[model][condition]=shares
            correct=int(np.trace(counts[labels].to_numpy()))
            audit.append({'model':model,'condition':condition,'objects':len(panel),'classes':len(labels),
                'correct':correct,'accuracy':correct/len(panel),'invalid_responses':int((~panel['_valid']).sum()),
                'omitted_prediction_count':int(len(panel)-counts[labels].to_numpy().sum())})
            for truth in labels:
                for predicted in counts.columns:
                    count=int(counts.loc[truth,predicted]); n=int(support[truth])
                    cells.append({'model':model,'condition':condition,'true_label':truth,'pred_label':predicted,
                                  'count':count,'support':n,'fraction':count/n,'percentage':100*count/n})
            columns=['sample_id','true_label','pred_label','status','view_count','repeat','source_run']
            selected=panel[[c for c in columns if c in panel]].copy()
            selected.insert(0,'condition',condition); selected.insert(0,'model',model)
            decisions.append(selected)
    pd.DataFrame(cells).to_csv(data/'S3DIS_confusion_counts.csv',index=False)
    pd.DataFrame(audit).to_csv(data/'S3DIS_confusion_validation.csv',index=False)
    pd.concat(decisions,ignore_index=True).to_csv(data/'S3DIS_object_predictions.csv',index=False)
    jev=next((item for item in items if item['definition']['model']=='typesafe/jev-1.14'),None)
    if jev:
        rgb,aabb=jev['panels']['rgb'],jev['panels']['rgb_aabb']
        paired=rgb[['sample_id','true_label','pred_label','status','source_run']].rename(columns={
            'pred_label':'pred_rgb','status':'status_rgb','source_run':'source_run_rgb'}).merge(
            aabb[['sample_id','true_label','pred_label','status','source_run']].rename(columns={
                'pred_label':'pred_rgb_aabb','status':'status_rgb_aabb','source_run':'source_run_rgb_aabb'}),
            on=['sample_id','true_label'],how='inner',validate='one_to_one')
        paired['correct_rgb']=paired.status_rgb.eq('ok') & paired.true_label.eq(paired.pred_rgb)
        paired['correct_rgb_aabb']=paired.status_rgb_aabb.eq('ok') & paired.true_label.eq(paired.pred_rgb_aabb)
        paired.sort_values('sample_id').to_csv(data/'Jev_S3DIS_objektweise_RGB_RGB_AABB.csv',index=False)
    def draw(chosen,individual=False):
        rows=len(chosen)
        fig,axes=plt.subplots(rows,2,figsize=(12.0,5.3) if individual else (10.6,4.45*rows),squeeze=False)
        for row,item in enumerate(chosen):
            definition=item['definition']
            for col,condition in enumerate(['rgb','rgb_aabb']):
                ax=axes[row,col]; matrix=prepared[definition['model']][condition]
                n=len(labels)
                im=ax.pcolormesh(np.arange(n+1)-.5,np.arange(n+1)-.5,matrix.to_numpy(),
                                 cmap='Blues',vmin=0,vmax=1,edgecolors='none',rasterized=False)
                ax.set(xlim=(-.5,n-.5),ylim=(n-.5,-.5),aspect='equal')
                ax.set_xticks(range(len(labels)),labels,rotation=90,fontsize=9)
                ax.set_yticks(range(len(labels)),labels if col==0 else ['']*len(labels),fontsize=9)
                ax.set_xlabel('Predicted class',fontsize=11,labelpad=7)
                if col==0:ax.set_ylabel('True class',fontsize=11)
                ax.set_title(definition['title']+' - '+('RGB' if col==0 else 'RGB + AABB'),fontsize=13,pad=11)
                ax.tick_params(length=3,width=.7)
                if individual:
                    ax.set_xticks(np.arange(-.5,len(labels)),minor=True)
                    ax.set_yticks(np.arange(-.5,len(labels)),minor=True)
                    ax.grid(which='minor',color='white',linewidth=.4)
                    ax.tick_params(which='minor',length=0)
                    for y,truth in enumerate(labels):
                        for x,predicted in enumerate(labels):
                            value=matrix.iloc[y,x]
                            if value>=.005 or truth==predicted:
                                ax.text(x,y,f'{100*value:.0f}%',ha='center',va='center',fontsize=8,
                                        color='white' if value>=.55 else '#14304b')
        fig.subplots_adjust(left=.105,right=.885,bottom=.18 if rows==1 else .053,
                            top=.88 if rows==1 else .974,wspace=.22,hspace=.66)
        color=fig.add_axes([.92,.24 if rows==1 else .16,.016,.50 if rows==1 else .69])
        colorbar=fig.colorbar(im,cax=color,format=PercentFormatter(1))
        if colorbar.solids is not None:
            colorbar.solids.set_rasterized(False)
            colorbar.solids.set_edgecolor('face')
        colorbar.set_label('True class',fontsize=11)
        return fig
    with plt.rc_context({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'}):
        fig=draw(items); fig.savefig(plots/f'{STEM}.pdf',bbox_inches='tight')
        fig.savefig(plots/f'{STEM}.png',dpi=dpi,bbox_inches='tight'); plt.close(fig)
        if jev:
            fig=draw([jev],True); fig.savefig(plots/'Jev_S3DIS_confusion_RGB_RGB_AABB.pdf',bbox_inches='tight')
            fig.savefig(plots/'Jev_S3DIS_confusion_RGB_RGB_AABB.png',dpi=dpi,bbox_inches='tight'); plt.close(fig)
    names=', '.join(item['definition']['title'] for item in items)
    n=len(items[0]['panels']['rgb'])
    caption=(f'S3DIS Area 5 confusion matrices for {names}, arranged by model from top to bottom. '
             f'RGB input is shown on the left and RGB+AABB input on the right. All panels use the same {n:,} objects and '
             f'{len(labels)} true classes. The common blue scale shows the proportion of saved hard classification decisions '
             'within each true class. Original true-class denominators are retained; invalid responses and predictions '
             'outside the displayed class set remain in the denominators but are not displayed as separate columns.')
    snippet=('\\clearpage\n\\begin{figure*}[p]\n\\centering\n'
        f'\\includegraphics[width=\\textwidth,height=0.82\\textheight,keepaspectratio]{{paper_figures/{STEM}.pdf}}\n'
        f'\\caption{{{caption}}}\n\\label{{fig:s3dis_confusion}}\n\\end{{figure*}}\n\\clearpage\n')
    (tex/'s3dis_confusion.tex').write_text(snippet,encoding='utf-8')
    return {'models':[i['definition']['model'] for i in items],'objects_per_condition':n,'classes':labels,
            'figure':f'paper_figures/{STEM}.pdf','decision_source':'saved pred_label, never probability argmax',
            'audit':audit}
