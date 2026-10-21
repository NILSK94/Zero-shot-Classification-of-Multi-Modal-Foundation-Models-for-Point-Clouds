# Multimodal Point Cloud Classifier

MMFM Lab renders isometric point cloud views, classifies objects with multimodal models, and evaluates saved results.

## Getting started

Use Python 3.12. On Windows, run `START_TOOL.bat` to set up the environment and start the application.

For a manual installation, create a virtual environment:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS, then run:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Interface

Choose dataset folders and models in the sidebar. Enter credentials there or set the environment variables listed in `.env.example`. The optional mock mode checks the pipeline without model requests.

- **Overview:** inspect available objects, classes and experimental configurations.
- **Prepare S3DIS:** generate views from the point clouds.
- **JEV image test:** inspect a request and its criteria, then classify one object using one or two views after confirming the API charges.
- **Experiments:** run ModelNet with one to four views and S3DIS with RGB or RGB+AABB. Each response is saved immediately.
- **Results:** inspect saved runs, automatic summaries, historical results and the optional matrix explorer.
- **Evaluations:** generate and download figures and data from saved results, without new model requests.


## Local data and settings

Datasets and results are not included in the source repository. The ignored `data_paths.local.json` can store local paths for `modelnet`, `s3dis`, `prepared`, `runs`, `legacy_results`, `historical_csv` and `paper_export`. Environment variables from `.env.example` take precedence. Evaluation does not modify source datasets.

`.gitignore` includes application source, required configuration and launch scripts. Local datasets, results, generated graphics, backups, machine-specific paths, virtual environments, credentials, private documents and development tests are excluded from publication.
