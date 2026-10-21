"""Paired S3DIS hard decisions and class comparisons for publication exports."""
from pathlib import Path
import numpy as np
import pandas as pd
from .s3dis_publication import validate_panels

CLASS_STEM='S3DIS_Klassenvergleich_RGB_RGB_AABB'
CHANGE_STEM='S3DIS_AABB_gepaarte_Klassifikationswechsel'
STATES=['correct_to_correct','incorrect_to_correct','correct_to_incorrect','incorrect_to_incorrect']


def paired_decisions(items: list[dict]) -> pd.DataFrame:
    """Match objects by ID and truth; retain invalid decisions as incorrect."""
    validate_panels(items)
    frames=[]
    for item in items:
        parts=[]
        for condition in ['rgb','rgb_aabb']:
            frame=item['panels'][condition]
            if frame.sample_id.isna().any() or frame.sample_id.astype(str).str.strip().eq('').any():
                raise ValueError('S3DIS object IDs must be nonempty.')
            columns=['sample_id','true_label','pred_label','_valid']+[c for c in ['status','source_run','repeat','view_count'] if c in frame]
            part=frame[columns].copy()
            part=part.rename(columns={c:f'{c}_{condition}' for c in columns if c not in ['sample_id','true_label']})
            parts.append(part)
        paired=parts[0].merge(parts[1],on=['sample_id','true_label'],validate='one_to_one',how='inner')
        if len(paired)!=len(parts[0]):raise ValueError('Incomplete paired S3DIS cohort.')
        paired.insert(0,'model',item['definition']['model'])
        paired.insert(1,'model_title',item['definition']['title'])
        for condition in ['rgb','rgb_aabb']:
            paired[f'correct_{condition}']=paired[f'_valid_{condition}'].astype(bool) & paired.true_label.eq(paired[f'pred_label_{condition}'])
        rgb,aabb=paired.correct_rgb,paired.correct_rgb_aabb
        paired[STATES[0]]=rgb & aabb
        paired[STATES[1]]=~rgb & aabb
        paired[STATES[2]]=rgb & ~aabb
        paired[STATES[3]]=~rgb & ~aabb
        effective=[paired[f'pred_label_{c}'].where(paired[f'_valid_{c}'].astype(bool),'[No valid response]') for c in ['rgb','rgb_aabb']]
        paired['prediction_changed']=effective[0].ne(effective[1])
        paired['incorrect_to_incorrect_changed']=paired.incorrect_to_incorrect & paired.prediction_changed
        frames.append(paired.sort_values('sample_id'))
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def paired_summary(paired: pd.DataFrame, by_class: bool=True) -> pd.DataFrame:
    keys=['model','model_title']+(['true_label'] if by_class else [])
    grouped=paired.groupby(keys,sort=False)
    columns=STATES+['correct_rgb','correct_rgb_aabb','prediction_changed','incorrect_to_incorrect_changed']
    result=grouped[columns].sum().astype(int)
    result.insert(0,'objects',grouped.size())
    if not result[STATES].sum(axis=1).equals(result.objects):raise ValueError('Paired states do not partition all objects.')
    result['net_correct']=result.incorrect_to_correct-result.correct_to_incorrect
    if not result.net_correct.equals(result.correct_rgb_aabb-result.correct_rgb):raise ValueError('Paired changes and accuracy differ.')
    for c in STATES+['correct_rgb','correct_rgb_aabb','net_correct']:
        result[c+'_percent']=100*result[c]/result.objects
    return result.reset_index()


def _escape(value: str) -> str:
    return value.replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')


def write_s3dis_paired_analysis(root: Path, items: list[dict], dpi: int=600, *, appendix_letter: str='C') -> dict:
    labels=validate_panels(items)
    if not labels:return {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import PercentFormatter
    plots=root/'paper_figures'; data=root/'paper_data'; tex=root/'paper_tex'
    for folder in [plots,data,tex]:folder.mkdir(parents=True,exist_ok=True)
    paired=paired_decisions(items); classes=paired_summary(paired); totals=paired_summary(paired,False)
    definitions=[i['definition'] for i in items]; models=[d['model'] for d in definitions]
    paired.to_csv(data/'S3DIS_AABB_object_pairs.csv',index=False)
    classes.to_csv(data/f'{CHANGE_STEM}.csv',index=False)
    totals.to_csv(data/'S3DIS_AABB_paired_summary.csv',index=False)
    support=paired[paired.model.eq(models[0])].groupby('true_label').size().reindex(labels).rename('objects').reset_index()
    support.to_csv(data/'S3DIS_class_support.csv',index=False)
    overview=[]
    for row in classes.to_dict('records'):
        for condition in ['rgb','rgb_aabb']:
            overview.append({k:row[k] for k in ['model','model_title','true_label','objects']} |
                {'condition':condition,'correct':row['correct_'+condition],'percentage':row['correct_'+condition+'_percent']})
    pd.DataFrame(overview).to_csv(data/f'{CLASS_STEM}.csv',index=False)
    styles={'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'}
    with plt.rc_context(styles),PdfPages(plots/'S3DIS_Klassen_und_AABB_Wechsel.pdf') as atlas:
        for stem,title,fields,titles,decimals in [
            (CLASS_STEM,'S3DIS: correct classifications by class',
             ['correct_rgb_percent','correct_rgb_aabb_percent'],['RGB','RGB + AABB'],0),
            (CHANGE_STEM,'S3DIS: paired classification changes with AABB',
             ['incorrect_to_correct_percent','correct_to_incorrect_percent'],['Incorrect → correct','Correct → incorrect'],1)]:
            fig=plt.figure(figsize=(12.2,8.0))
            grid=fig.add_gridspec(1,2,left=.115,right=.89,bottom=.21,top=.83,wspace=.13)
            fig.suptitle(title,fontsize=18,y=.975)
            fig.text(.5,.924,'4 views' if stem==CLASS_STEM else 'RGB → RGB + AABB | 4 views',ha='center',fontsize=12)
            for col,(field,panel_title) in enumerate(zip(fields,titles)):
                ax=fig.add_subplot(grid[0,col])
                matrix=classes.pivot(index='true_label',columns='model',values=field).reindex(index=labels,columns=models)
                im=ax.pcolormesh(np.arange(len(models)+1)-.5,np.arange(len(labels)+1)-.5,matrix.to_numpy(),
                    cmap='Blues',vmin=0,vmax=100,edgecolors='white',linewidth=.5,rasterized=False)
                ax.set(xlim=(-.5,len(models)-.5),ylim=(len(labels)-.5,-.5))
                ax.set_xticks(range(len(models)),[d['title'] for d in definitions],rotation=32,ha='right',fontsize=10)
                ax.set_yticks(range(len(labels)),labels,fontsize=10)
                ax.tick_params(length=0,pad=5)
                if col:ax.tick_params(axis='y',labelleft=False)
                else:ax.set_ylabel('True class',fontsize=12)
                ax.set_title(panel_title,fontsize=14,pad=12)
                for y,x in np.ndindex(matrix.shape):
                    value=matrix.iloc[y,x]
                    ax.text(x,y,f'{value:.{decimals}f}%',ha='center',va='center',fontsize=10,
                        color='white' if value>=55 else '#14304b')
            bar=fig.colorbar(im,cax=fig.add_axes([.92,.30,.014,.43]),ticks=[0,20,40,60,80,100],format=PercentFormatter(100,decimals=0))
            bar.set_label('True class',fontsize=11)
            if bar.solids is not None:bar.solids.set_rasterized(False);bar.solids.set_edgecolor('face')
            fig.savefig(plots/f'{stem}.png',dpi=dpi,bbox_inches='tight')
            fig.savefig(plots/f'{stem}.pdf',bbox_inches='tight');atlas.savefig(fig,bbox_inches='tight');plt.close(fig)
    n=int(totals.iloc[0].objects)
    caption=(f'Paired classification changes from RGB to RGB+AABB on the same {n:,} S3DIS Area 5 objects, using four views in both conditions. '
        'Left: previously incorrect decisions that become correct. Right: previously correct decisions that become incorrect. '
        'Every cell is a percentage of all evaluated objects in that true class, rather than a conditional correction or error rate. '
        'The net class-level change in percentage points is the left value minus the right value before rounding. '
        'Objects correct in both conditions and objects incorrect in both conditions are not shown in the panels, but remain in every denominator. '
        'All classes are shown, with no selection by effect size. Invalid responses count as incorrect. '
        'These are observed paired changes from one run per condition, not repeated-run estimates of a causal effect. '
        'Class supports, all four transition counts and the RGB/AABB class comparison are provided in Appendix~\\ref{app:s3dis_paired}.')
    snippet=('\\begin{figure*}[ht]\n\\centering\n'
        f'\\includegraphics[width=\\textwidth]{{paper_figures/{CHANGE_STEM}.pdf}}\n'
        f'\\caption{{{caption}}}\n\\label{{fig:s3dis_aabb_paired_changes}}\n\\end{{figure*}}\n')
    (tex/'s3dis_paired_changes.tex').write_text(snippet,encoding='utf-8')
    if appendix_letter not in ['B','C']:raise ValueError('Unsupported S3DIS appendix letter.')
    # Appendix C follows ModelNet A and Jev B; without Jev, use B.
    appendix=[r'\clearpage',r'\setcounter{figure}{0}',r'\renewcommand{\thefigure}{C\arabic{figure}}',
        r'\ifdefined\theHfigure\renewcommand{\theHfigure}{appendix.C.\arabic{figure}}\fi',
        r'\setcounter{table}{0}',r'\renewcommand{\thetable}{C\arabic{table}}',
        r'\ifdefined\theHtable\renewcommand{\theHtable}{appendix.C.\arabic{table}}\fi',
        r'\section{Paired S3DIS Classification with AABB}',r'\label{app:s3dis_paired}',
        'The conditions are paired by saved object ID and ground-truth class. All models use the same objects. '
        'The following class comparison uses final saved labels, independently of the probability analysis.\n',
        r'\begin{figure*}[p]',r'\centering',
        f'\\includegraphics[width=\\textwidth]{{paper_figures/{CLASS_STEM}.pdf}}',
        r'\caption{Percentage of correctly classified objects in each S3DIS true class with RGB (left) and RGB+AABB (right). '
        f'Four identical views are used in both conditions. All {len(labels)} classes are shown. Cell values and the common blue scale are percentages of all evaluated objects in the corresponding true class.'+'}',
        r'\label{fig:s3dis_class_comparison}',r'\end{figure*}',r'\clearpage',
        r'\begin{table*}[ht]',r'\centering',r'\small',
        r'\caption{Paired correctness transitions from RGB to RGB+AABB. Counts in the four transition columns partition the full evaluated cohort for each model. Net is incorrect-to-correct minus correct-to-incorrect; the accuracy change uses all objects as denominator.}',
        r'\label{tab:s3dis_paired_summary}',r'\begin{tabular}{lrrrrrr}',r'\hline',
        r'Model & Right--right & Wrong--right & Right--wrong & Wrong--wrong & Net & $\Delta$ (pp) \\',r'\hline']
    for model in models:
        row=totals[totals.model.eq(model)].iloc[0]
        appendix.append(_escape(row.model_title)+' & '+' & '.join(str(int(row[c])) for c in STATES+['net_correct'])+
            f' & {row.net_correct_percent:.2f} '+r'\\')
    appendix.extend([r'\hline',r'\end{tabular}',r'\end{table*}',
        r'\begin{table}[ht]',r'\centering',r'\caption{Evaluated objects per S3DIS class. These same denominators are used for every model and both input conditions. Small classes, especially beam, should be interpreted with their sample counts in mind.}',
        r'\label{tab:s3dis_class_support}',r'\begin{tabular}{lr}',r'\hline',r'True class & Objects \\',r'\hline'])
    appendix.extend(_escape(row.true_label)+f' & {row.objects} '+r'\\' for row in support.itertuples())
    appendix.extend([r'\hline',f'Total & {n} '+r'\\',r'\hline',r'\end{tabular}',r'\end{table}',r'\clearpage'])
    appendix_text='\n'.join(appendix)+'\n'
    if appendix_letter!='C':appendix_text=appendix_text.replace('C\\arabic','B\\arabic').replace('appendix.C.','appendix.B.')
    (tex/'appendix_s3dis.tex').write_text(appendix_text,encoding='utf-8')
    results=['Figure~\\ref{fig:s3dis_aabb_paired_changes} separates corrected errors from newly introduced errors on the same objects.']
    for model in models:
        row=totals[totals.model.eq(model)].iloc[0]
        results.append(f'For {_escape(row.model_title)}, {int(row.incorrect_to_correct)} incorrect decisions become correct and '
            f'{int(row.correct_to_incorrect)} correct decisions become incorrect, yielding a net change of '
            f'{int(row.net_correct):+d} correct objects ({row.net_correct_percent:+.2f} percentage points).')
    results.append('Both-direction changes can therefore be larger than the net accuracy change suggests. '
        'The class comparison, all four transition totals and class supports are reported in '
        'Appendix~\\ref{app:s3dis_paired}, Figure~\\ref{fig:s3dis_class_comparison} and '
        'Tables~\\ref{tab:s3dis_paired_summary} and~\\ref{tab:s3dis_class_support}. '
        'These descriptive paired outcomes do not quantify variation across repeated runs.\n')
    (tex/'s3dis_paired_results.tex').write_text(' '.join(results)+'\n'+r'\input{paper_tex/s3dis_paired_changes}'+'\n',encoding='utf-8')
    return {'objects_per_model':n,'classes':labels,'models':models,'class_figure':f'paper_figures/{CLASS_STEM}.pdf',
        'paired_figure':f'paper_figures/{CHANGE_STEM}.pdf','normalization':'all evaluated objects within the true class',
        'states':STATES,'summary':totals.to_dict('records'),'pairing':'exact sample_id and true_label; saved pred_label; invalid counts incorrect'}
