"""Portable figures and data exports without manuscript instructions or LaTeX."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,tempfile,zipfile
from .publication_export import build_publication_export,PublicationExport


def build_evaluation_export(runs_root: str | Path, output_root: str | Path, *, repeat: int=1,
    profile_path: str | Path | None=None, reference_figures: str | Path | None=None,
    png_dpi: int=600, historical_csv_root: str | Path | None=None) -> PublicationExport:
    """Reuse validated numerical renderers; publish only graphics, data and provenance."""
    output=Path(output_root).expanduser().resolve();runs=Path(runs_root).expanduser().resolve()
    if output==runs or runs in output.parents:
        raise ValueError('The figure export must not be placed inside the results-run folder.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='evaluation_export_',dir=output.parent) as folder:
        temp=Path(folder)
        source=build_publication_export(runs,temp/'source',repeat=repeat,profile_path=profile_path,
            reference_figures=reference_figures,png_dpi=png_dpi,historical_csv_root=historical_csv_root)
        stage=temp/'evaluation';stage.mkdir()
        members={}
        for relative in source.manifest['files']:
            path=Path(relative)
            if relative.startswith('paper_figures/') and path.suffix.lower() in ['.png','.pdf','.jpg','.jpeg','.eps']:
                name='figures/'+relative.removeprefix('paper_figures/')
            elif relative.startswith('paper_data/') and path.suffix.lower() in ['.csv','.json']:
                name='data/'+relative.removeprefix('paper_data/')
            elif relative=='paper_figure_profile.json':name='figure_profile.json'
            else:continue
            target=stage/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source.output_dir/relative,target)
            if name.endswith('_provenance.json'):
                info=json.loads(target.read_text(encoding='utf-8'))
                info=json.loads(json.dumps(info).replace('paper_figures/','figures/').replace('paper_data/','data/'))
                if isinstance(info.get('files'),dict):
                    info['files']={key:digest for key,digest in info['files'].items() if key.startswith(('figures/','data/'))}
                target.write_text(json.dumps(info,indent=2),encoding='utf-8')
            members[name]=hashlib.sha256(target.read_bytes()).hexdigest()
        # Carry numerical provenance without manuscript labels, appendix order or instructions.
        keys=['profile_version','generator_sha256','repeat','png_dpi','runs','models','s3dis_models',
            'selection_warnings','excluded_panels','invalid_responses','macro_f1','selections']
        manifest={key:source.manifest[key] for key in keys}
        manifest.update(format_version=1,created_utc=datetime.now(timezone.utc).isoformat(),export_kind='evaluation')
        manifest['generator_sha256']['evaluation_export.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        for key in ['s3dis_confusions','s3dis_paired_analysis','pilot_figure7','reference_assets']:
            if key in source.manifest:
                value=json.loads(json.dumps(source.manifest[key]).replace('paper_figures/','figures/').replace('paper_data/','data/'))
                if isinstance(value,dict) and 'files' in value:
                    value['files']={name:digest for name,digest in value['files'].items() if name in members}
                manifest[key]=value
        probabilities=source.manifest.get('jev_probabilities',{})
        manifest['jev_probabilities']={k:v for k,v in probabilities.items() if k!='figures'}
        manifest['jev_probabilities']['figures']=[{'file':f'figures/{item["stem"]}.png'} for item in probabilities.get('figures',[])]
        manifest['files']=members
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        output.mkdir(parents=True,exist_ok=True)
        previous=output/'manifest.json'
        if previous.is_file():
            old=json.loads(previous.read_text(encoding='utf-8'))
            for relative in set(old.get('files',{}))-set(members):
                path=(output/relative).resolve()
                if output in path.parents and path.is_file():path.unlink()
        for relative in [*members,'manifest.json']:
            target=(output/relative).resolve()
            if output not in target.parents:raise ValueError('Invalid export path.')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(stage/relative,target)
    archive_path=output/'MMFM_Evaluations.zip'
    with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted([*manifest['files'],'manifest.json']):archive.write(output/relative,relative)
    return PublicationExport(output,archive_path,manifest)
