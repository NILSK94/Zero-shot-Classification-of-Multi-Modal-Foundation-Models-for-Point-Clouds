"""Reproducible figure and data downloads for the evaluation interface."""
from pathlib import Path
import json
import streamlit as st

from .publication_export import load_profile
from .evaluation_export import build_evaluation_export


def render_publication_export(runs_root: Path, output_root: Path, *, reference_figures: Path | None = None,
                              historical_csv_root: Path | None = None) -> None:
    try:
        profile=load_profile()
    except (OSError,ValueError) as exc:
        st.error('The figure profile is missing or invalid. Include mmfm_tool/paper_figure_profile.json with the application. Details: '+str(exc))
        return
    with st.expander('Export evaluations',expanded=True):
        st.write('Generates focused and complete matrices, ModelNet and S3DIS overviews, paired AABB classification changes, and Jev probability figures. Uses saved results only, with no API charges.')
        st.caption('Consistent colours and class selection · Percentages · Original denominators retained · PDF and PNG')
        first,second=st.columns([1,3])
        repeat=first.number_input('Repetition',min_value=1,value=1,step=1,key='publication_repeat')
        destination=Path(second.text_input('Output folder',value=str(output_root),key='publication_output')).expanduser()
        reference=st.text_input('Include additional figures (optional)',
                                value=str(reference_figures) if reference_figures else '',key='publication_reference')
        historical=st.text_input('Historical pilot CSV archive (optional)',
                                value=str(historical_csv_root) if historical_csv_root and historical_csv_root.is_dir() else '',key='publication_historical')
        with st.expander('Fixed class selection'):
            for model in profile['models']:
                improved=', '.join(c.replace('_',' ') for c in model['gains'])
                difficult=', '.join(c.replace('_',' ') for c in model['difficult'])
                st.write(f'**{model["title"]}:** Improvements: {improved}. Difficult classes: {difficult}.')
            st.caption('The selection highlights changes and persistent difficulties. Overall metrics retain all objects. Frequent prediction destinations are added automatically; remaining predictions are grouped as Other classes.')
        if st.button('Generate evaluations',key='publication_generate',type='primary'):
            try:
                with st.spinner('Validating saved results and generating figures …'):
                    result=build_evaluation_export(runs_root,destination,repeat=int(repeat),
                                                    reference_figures=reference.strip() or None,historical_csv_root=historical.strip() or None)
                st.success(f'Evaluations saved: {result.output_dir}')
            except (OSError,ValueError,KeyError) as exc:
                st.error(f'The graphics package was not created: {exc}')
        manifest_path=destination/'manifest.json'
        if manifest_path.is_file():
            try:
                manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
                st.caption(f'Profile: {manifest["profile_version"]} · Repetition {manifest["repeat"]} · '
                           f'{len(manifest["models"])} ModelNet models · {len(manifest["s3dis_models"])} S3DIS models')
                if manifest.get('profile_version')!=profile['version']:
                    st.warning('The saved figures use a different profile version. Regenerate the package to apply the current figures and labels.')
                figure_root=destination/('figures' if (destination/'figures').is_dir() else 'paper_figures')
                data_root=destination/('data' if (destination/'data').is_dir() else 'paper_data')
                for warning in manifest.get('selection_warnings',[]): st.warning(warning)
                pilot_png=figure_root/'Pilot_Figure7_confusion_matrices.png'
                if pilot_png.is_file():
                    with st.expander('Pilot evaluation'):
                        st.caption('The fixed layout contains all twelve historical panels. Stable Diffusion reproduces the original caption/synonym matching. ModelNet cells reproduce the archived CSV row counts; multiple views are not independent objects.')
                        st.download_button('Download pilot evaluation as PNG',pilot_png.read_bytes(),
                            file_name=pilot_png.name,mime='image/png',key='publication_pilot_png')
                with st.expander('S3DIS: confusion matrices including Jev'):
                    st.write('RGB and RGB+AABB use the same object cohort. These matrices count saved classification decisions; probabilities are analysed separately.')
                    picture=figure_root/'s3dis_confusion_rgb_aabb_4models.png'
                    if picture.is_file(): st.image(str(picture),width='stretch')
                    csv=data_root/'Jev_S3DIS_objektweise_RGB_RGB_AABB.csv'
                    if csv.is_file():
                        st.download_button('Jev: download ground truth and both predictions',csv.read_bytes(),
                            file_name=csv.name,mime='text/csv',key='publication_jev_s3dis_decisions')
                with st.expander('S3DIS: class comparison and paired AABB changes'):
                    info=manifest.get('s3dis_paired_analysis',{})
                    if info:
                        st.write('Objects are paired by ID. Left: incorrect → correct. Right: correct → incorrect. Percentages use all objects in the corresponding true class as denominator. Their difference gives the net change in percentage points. All classes are shown.')
                        st.dataframe([{'Model':row['model_title'],'Objects':row['objects'],
                            'Incorrect → correct':row['incorrect_to_correct'],'Correct → incorrect':row['correct_to_incorrect'],
                            'Net correct objects':row['net_correct'],
                            'Accuracy change (pp)':round(row['net_correct_percent'],2)} for row in info['summary']],hide_index=True)
                        for stem in ['S3DIS_AABB_gepaarte_Klassifikationswechsel','S3DIS_Klassenvergleich_RGB_RGB_AABB']:
                            image=figure_root/f'{stem}.png'
                            if image.is_file():
                                st.image(str(image),width='stretch')
                                st.download_button('Download paired changes as PNG' if 'Klassifikationswechsel' in stem else 'Download class comparison as PNG',image.read_bytes(),file_name=image.name,
                                    mime='image/png',key='publication_'+stem)
                        st.caption('The four correctness states partition all object pairs. Each condition has one run; variability across repeated runs is not estimated. Interpret small classes alongside their sample counts.')
                    else:
                        st.info('This saved package does not contain paired S3DIS analysis yet. Regenerate the package.')
                probability_info=manifest.get('jev_probabilities',{})
                with st.expander('Jev: saved class probabilities'):
                    st.write('The matrices show mean class probabilities. Correctness plots compare the probability of the chosen class with observed correctness. The separate API confidence field is not used. Rounded totals of 99% are retained without renormalisation.')
                    conditions=probability_info.get('conditions',[])
                    if conditions:
                        st.dataframe([{'Dataset':'ModelNet' if s['dataset']=='modelnet' else 'S3DIS',
                            'Input':s['condition'].replace('views_','Views: ').replace('rgb_aabb','RGB + AABB').replace('rgb','RGB'),
                            'Objects':s['objects_expected'],'Valid vectors':s['vectors_valid'],
                            'Coverage (%)':round(100*s['coverage'],1),
                            'ECE (pp)':round(100*s['ece_10_bins'],2) if s.get('ece_10_bins') is not None else None,
                            'Rounded vectors':s['rounded_nonunit_vectors'],
                            'Choice below maximum':s['choice_not_argmax']} for s in conditions],hide_index=True)
                        st.caption('ECE uses ten fixed probability bins. These values describe the selected runs and do not establish model calibration. Complete values and exclusions are included in the package.')
                        for name in ['Jev_ModelNet_probability_focus','Jev_probability_correctness','Jev_s3dis_probability_full']:
                            image=figure_root/f'{name}.png'
                            if image.is_file(): st.image(str(image),width='stretch')
                    else:
                        st.info('No matching Jev Choice vectors are available for this repetition. Independent historical Noul scores are not used as a class distribution.')
                with st.expander('Source data and excluded conditions'):
                    st.json({'modelnet':manifest['models'],'s3dis':manifest['s3dis_models'],
                             'excluded':manifest['excluded_panels'],'selections':manifest['selections']})
                zip_path=destination/'MMFM_Evaluations.zip'
                if not zip_path.is_file():zip_path=destination/'MMFM_Auswertungen.zip'
                if zip_path.is_file():
                    st.download_button('Download figures and data as ZIP',zip_path.read_bytes(),
                                        file_name='MMFM_Evaluations.zip',mime='application/zip',key='publication_zip')
                with st.expander('Download an individual figure'):
                    available=sorted(Path(name).name for name in manifest.get('files',{})
                        if name.startswith(('figures/','paper_figures/')) and name.endswith('.png')
                        and '/reference/' not in name and (destination/name).is_file())
                    if available:
                        filename=st.selectbox('Select a figure',available,key='publication_single_file')
                        image=figure_root/filename
                        st.image(str(image),width='stretch')
                        png_column,pdf_column=st.columns(2)
                        png_column.download_button('Download PNG',image.read_bytes(),
                            file_name=image.name,mime='image/png',key='publication_single_png')
                        pdf=image.with_suffix('.pdf')
                        if pdf.is_file():
                            pdf_column.download_button('Download PDF',pdf.read_bytes(),
                                file_name=pdf.name,mime='application/pdf',key='publication_single_pdf')
                with st.expander('Preview generated figures'):
                    choices={model['model']:model['stem'] for model in profile['models'] if model['model'] in manifest['models']}
                    if choices:
                        selected=st.selectbox('Preview model',list(choices),key='publication_preview_model')
                        for suffix in ['_confusion_focus.png','_confusion.png']:
                            path=figure_root/(choices[selected]+suffix)
                            if path.is_file(): st.image(str(path),width='stretch')
            except (OSError,ValueError,KeyError) as exc:
                st.warning(f'Could not read the saved export package: {exc}')
