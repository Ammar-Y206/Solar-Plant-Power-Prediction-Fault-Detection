# Samsung SIC Graduation Project

This repository contains the organized deliverables for the Samsung SIC graduation project. It includes two complementary machine-learning pipelines for solar-plant forecasting and anomaly detection.

## Repository structure

```text
projects/
├── anomaly-detection/
│   ├── anomaly_detection_final.ipynb
│   ├── preprocessing_and_feature_engineering.ipynb
│   ├── phase_a/                    # processed Phase-A inputs and metadata
│   ├── phase_b/                    # Phase-B training outputs
│   ├── model_artifacts/            # trained deployment model and calibration
│   ├── predict.py                  # production prediction wrapper
│   ├── test_smoke.py
│   ├── sample_input*.csv
│   ├── expected_output*.csv
│   ├── requirements.txt
│   └── README.md
└── forecasting/
    ├── forecasting_training_pipeline.ipynb
    ├── forecast_inputs/
    ├── artifacts/                  # generated outputs are created here
    ├── f2_config.json
    ├── f3_config.json
    ├── f4_config.json
    ├── requirements.txt
    └── README.md
```

## Quick start

Create a virtual environment and install the dependencies for the pipeline you want to run:

```bash
cd projects/anomaly-detection
python -m venv .venv
.venv/bin/pip install -r requirements.txt
pytest -q test_smoke.py
```

For the forecasting pipeline:

```bash
cd projects/forecasting
python -m venv .venv
.venv/bin/pip install -r requirements.txt
jupyter lab
```

The detailed run instructions and protocol notes are in the README inside each project folder.

## GitHub upload note

The repository contains large CSV inputs required by the notebooks. Upload this folder as a Git repository rather than uploading the generated ZIP as a single GitHub file. The ZIP is provided as a convenient backup/package; GitHub will not display its internal folders as repository files.
