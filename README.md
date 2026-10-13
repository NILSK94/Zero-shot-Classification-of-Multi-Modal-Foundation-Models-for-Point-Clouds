# Multimodal Point Cloud Classifier

Code for *An Evaluation of Zero-shot Classification of Multi-Modal Foundation Models for Point Clouds*.

Convert point clouds to isometric images, classify them, and compare the results.

## Start

Use Python 3.12. On Windows, run `START_TOOL.bat`.

On other systems:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Set your dataset paths, model IDs, proxy URL, and API keys in the sidebar. The mock model checks the pipeline without API calls.

## Data and evaluation

- ModelNet: compare 1–4 views of the same test objects. The paper cohort has 1,860 objects and 33 classes.
- S3DIS: compare RGB and RGB + bounding-box dimensions on the same instances. Prepare views before running an experiment.
- Report end-to-end accuracy and coverage. Invalid or missing responses count as incorrect.

No datasets or results are included. Set your own paths in the sidebar. Optional environment variables are listed in `.env.example`; export them in your shell before starting the app.

Runs, figures, and paper exports are saved in `artifacts/`. Local data, API keys, and generated files are excluded from Git.

## Files

- `app.py`, `mmfm_tool/`: experiment and evaluation tool.
- `converter_OFF_PLY.py`, `converter_PLY_PNG.py`: mesh conversion and rendering. Set input and output paths before use.
- `*_classifier_*.ipynb`: original classification notebooks. Set paths and API keys before use.
- `Confusion_Matrix_Evaluation.ipynb`, `Evaluator_stablediffusion.py`: original evaluation scripts.

Change class lists and view angles in `mmfm_tool/constants.py`. The original scripts and notebooks use separate dependencies listed in `requirements-legacy.txt`.

## Tests

```sh
python -m pytest -q
```
