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
        st.error('Das Grafikprofil fehlt oder ist ungültig. Bitte mmfm_tool/paper_figure_profile.json '
                 'zusammen mit dem Programm übernehmen. Details: '+str(exc))
        return
    with st.expander('Auswertungen exportieren',expanded=True):
        st.write('Erzeugt die abgestimmten Detailmatrizen, vollständigen Matrizen, '
                 'ModelNet- und S3DIS-Übersichten, gepaarte AABB-Klassifikationswechsel und Jev-Wahrscheinlichkeitsgrafiken. '
                 'Nur gespeicherte Ergebnisse werden gelesen; es entstehen keine API-Kosten.')
        st.caption('Feste Farben und Klassenauswahl · Prozentwerte · Originalnenner bleiben erhalten · PDF und PNG')
        first,second=st.columns([1,3])
        repeat=first.number_input('Wiederholung',min_value=1,value=1,step=1,key='publication_repeat')
        destination=Path(second.text_input('Ausgabeordner',value=str(output_root),key='publication_output')).expanduser()
        reference=st.text_input('Zusätzliche Abbildungen mitgeben (optional)',
                                value=str(reference_figures) if reference_figures else '',key='publication_reference')
        historical=st.text_input('Historisches Pilot-CSV-Archiv (optional)',
                                value=str(historical_csv_root) if historical_csv_root and historical_csv_root.is_dir() else '',key='publication_historical')
        with st.expander('Abgestimmte Klassenauswahl'):
            for model in profile['models']:
                improved=', '.join(c.replace('_',' ') for c in model['gains'])
                difficult=', '.join(c.replace('_',' ') for c in model['difficult'])
                st.write(f'**{model["title"]}:** Verbesserungen: {improved}. Schwierige Klassen: {difficult}.')
            st.caption('Die Auswahl illustriert die Ergebnisdiskussion. Alle Objekte bleiben in der Gesamtauswertung. '
                       'Weitere häufige vorhergesagte Klassen werden automatisch ergänzt; der Rest steht in Other classes.')
        if st.button('Auswertungen erzeugen',key='publication_generate',type='primary'):
            try:
                with st.spinner('Gespeicherte Ergebnisse prüfen und Grafiken erzeugen …'):
                    result=build_evaluation_export(runs_root,destination,repeat=int(repeat),
                                                    reference_figures=reference.strip() or None,historical_csv_root=historical.strip() or None)
                st.success(f'Auswertungen gespeichert: {result.output_dir}')
            except (OSError,ValueError,KeyError) as exc:
                st.error(f'Das Grafikpaket wurde nicht erstellt: {exc}')
        manifest_path=destination/'manifest.json'
        if manifest_path.is_file():
            try:
                manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
                st.caption(f'Profil: {manifest["profile_version"]} · Wiederholung {manifest["repeat"]} · '
                           f'{len(manifest["models"])} ModelNet-Modelle · {len(manifest["s3dis_models"])} S3DIS-Modelle')
                if manifest.get('profile_version')!=profile['version']:
                    st.warning('Die gespeicherten Grafiken stammen aus einem anderen Grafikstand. '
                               'Bitte das Paket neu erzeugen, um die aktuellen Abbildungen und Beschriftungen zu übernehmen.')
                figure_root=destination/('figures' if (destination/'figures').is_dir() else 'paper_figures')
                data_root=destination/('data' if (destination/'data').is_dir() else 'paper_data')
                for warning in manifest.get('selection_warnings',[]): st.warning(warning)
                pilot_png=figure_root/'Pilot_Figure7_confusion_matrices.png'
                if pilot_png.is_file():
                    with st.expander('Pilot-Auswertung'):
                        st.caption('Die feste Tafel enthält alle zwölf historischen Panels. Stable Diffusion wird nach der '
                                   'damaligen Caption-/Synonym-Prüfung dargestellt. ModelNet-Zellwerte reproduzieren die archivierten CSV-Zeilen; '
                                   'mehrere Ansichten sind keine zusätzlichen unabhängigen Objekte.')
                        st.download_button('Pilot-Auswertung als PNG herunterladen',pilot_png.read_bytes(),
                            file_name=pilot_png.name,mime='image/png',key='publication_pilot_png')
                with st.expander('S3DIS: Konfusionsmatrizen inklusive Jev'):
                    st.write('RGB und RGB+AABB auf derselben Objektbasis. Diese Matrizen zählen die gespeicherten '
                             'Klassifikationsentscheidungen; die Wahrscheinlichkeitsanalyse folgt separat.')
                    picture=figure_root/'s3dis_confusion_rgb_aabb_4models.png'
                    if picture.is_file(): st.image(str(picture),width='stretch')
                    csv=data_root/'Jev_S3DIS_objektweise_RGB_RGB_AABB.csv'
                    if csv.is_file():
                        st.download_button('Jev: Ground Truth und beide Vorhersagen herunterladen',csv.read_bytes(),
                            file_name=csv.name,mime='text/csv',key='publication_jev_s3dis_decisions')
                with st.expander('S3DIS: Klassenvergleich und gepaarte Wechsel durch AABB'):
                    info=manifest.get('s3dis_paired_analysis',{})
                    if info:
                        st.write('Dieselben Objekte werden über ihre IDs gepaart. Links stehen falsch → richtig, '
                                 'rechts richtig → falsch. Prozentwerte verwenden alle Objekte der jeweiligen wahren Klasse als Nenner. '
                                 'Die Differenz ergibt die Nettoänderung in Prozentpunkten. Es werden alle Klassen gezeigt.')
                        st.dataframe([{'Modell':row['model_title'],'Objekte':row['objects'],
                            'Falsch → richtig':row['incorrect_to_correct'],'Richtig → falsch':row['correct_to_incorrect'],
                            'Netto richtige Objekte':row['net_correct'],
                            'Accuracy-Änderung (pp)':round(row['net_correct_percent'],2)} for row in info['summary']],hide_index=True)
                        for stem in ['S3DIS_AABB_gepaarte_Klassifikationswechsel','S3DIS_Klassenvergleich_RGB_RGB_AABB']:
                            image=figure_root/f'{stem}.png'
                            if image.is_file():
                                st.image(str(image),width='stretch')
                                st.download_button('PNG herunterladen: '+stem,image.read_bytes(),file_name=image.name,
                                    mime='image/png',key='publication_'+stem)
                        st.caption('Die vier Richtigkeitszustände partitionieren alle Objektpaare. Ein Lauf je Bedingung; '
                                   'keine Aussage zur Streuung über Wiederholungen. Kleine Klassen anhand der Klassengrößen beurteilen.')
                    else:
                        st.info('Dieses gespeicherte Paket enthält noch keine gepaarte S3DIS-Analyse. Paket neu erzeugen.')
                probability_info=manifest.get('jev_probabilities',{})
                with st.expander('Jev: gespeicherte Klassenwahrscheinlichkeiten'):
                    st.write('Die Matrizen zeigen mittlere Klassenwahrscheinlichkeiten. Die Entscheidungsdiagramme vergleichen '
                             'die Wahrscheinlichkeit der gewählten Klasse mit der tatsächlichen Richtigkeit. '
                             'Das separate API-Feld confidence wird dafür nicht verwendet. '
                             'Gerundete Summen von 99 % bleiben erhalten; es wird nicht nachträglich normalisiert.')
                    conditions=probability_info.get('conditions',[])
                    if conditions:
                        st.dataframe([{'Datensatz':'ModelNet' if s['dataset']=='modelnet' else 'S3DIS',
                            'Eingabe':s['condition'].replace('views_','Ansichten: ').replace('rgb_aabb','RGB + AABB').replace('rgb','RGB'),
                            'Objekte':s['objects_expected'],'Gültige Vektoren':s['vectors_valid'],
                            'Abdeckung (%)':round(100*s['coverage'],1),
                            'ECE (pp)':round(100*s['ece_10_bins'],2) if s.get('ece_10_bins') is not None else None,
                            'Gerundete Vektoren':s['rounded_nonunit_vectors'],
                            'Wahl unter Maximum':s['choice_not_argmax']} for s in conditions],hide_index=True)
                        st.caption('ECE verwendet zehn feste Wahrscheinlichkeitsintervalle. Die Werte beschreiben diese Läufe; '
                                   'sie belegen keine Kalibrierung des Modells. Vollständige Werte und Ausschlüsse stehen im Paket.')
                        for name in ['Jev_ModelNet_probability_focus','Jev_probability_correctness','Jev_s3dis_probability_full']:
                            image=figure_root/f'{name}.png'
                            if image.is_file(): st.image(str(image),width='stretch')
                    else:
                        st.info('Für die ausgewählte Wiederholung liegen keine passenden Jev-Choice-Vektoren vor. '
                                'Unabhängige historische Noul-Scores werden nicht als Klassenverteilung verwendet.')
                with st.expander('Verwendete Daten und ausgeschlossene Bedingungen'):
                    st.json({'modelnet':manifest['models'],'s3dis':manifest['s3dis_models'],
                             'excluded':manifest['excluded_panels'],'selections':manifest['selections']})
                zip_path=destination/'MMFM_Auswertungen.zip'
                if zip_path.is_file():
                    st.download_button('Grafiken und Daten als ZIP herunterladen',zip_path.read_bytes(),
                                        file_name=zip_path.name,mime='application/zip',key='publication_zip')
                with st.expander('Einzelne Grafik herunterladen'):
                    available=sorted(Path(name).name for name in manifest.get('files',{})
                        if name.startswith(('figures/','paper_figures/')) and name.endswith('.png')
                        and '/reference/' not in name and (destination/name).is_file())
                    if available:
                        filename=st.selectbox('Grafik auswählen',available,key='publication_single_file')
                        image=figure_root/filename
                        st.image(str(image),width='stretch')
                        png_column,pdf_column=st.columns(2)
                        png_column.download_button('PNG herunterladen',image.read_bytes(),
                            file_name=image.name,mime='image/png',key='publication_single_png')
                        pdf=image.with_suffix('.pdf')
                        if pdf.is_file():
                            pdf_column.download_button('PDF herunterladen',pdf.read_bytes(),
                                file_name=pdf.name,mime='application/pdf',key='publication_single_pdf')
                with st.expander('Vorschau der erzeugten Abbildungen'):
                    choices={model['model']:model['stem'] for model in profile['models'] if model['model'] in manifest['models']}
                    if choices:
                        selected=st.selectbox('Modell der Vorschau',list(choices),key='publication_preview_model')
                        for suffix in ['_confusion_focus.png','_confusion.png']:
                            path=figure_root/(choices[selected]+suffix)
                            if path.is_file(): st.image(str(path),width='stretch')
            except (OSError,ValueError,KeyError) as exc:
                st.warning(f'Das gespeicherte Exportpaket ist nicht lesbar: {exc}')
