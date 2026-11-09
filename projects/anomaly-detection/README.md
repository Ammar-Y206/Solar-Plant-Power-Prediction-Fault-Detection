# Anomaly Detection Complete Project

This package contains the full solar plant anomaly-detection project. The project has three stages:

1. **Phase A — preprocessing and feature engineering:** reads the raw weather, plant, and array CSVs, builds UTC-indexed features and quality flags, creates event metadata, defines scoreable and clean rows, and writes the Phase-A tables.
2. **Phase B — model training:** trains and evaluates the existing constant baseline, LightGBM GBM, MLP, and autoencoder models using the Phase-A tables.
3. **Final report and deployment:** compares the forecasting models, calibrates the GBM residual detector, applies the rolling anomaly score and threshold, and provides the production `predict.py` wrapper.

## Package layout

```text
anomaly_detection_complete_project/
├── anomaly_detection_final.ipynb
├── preprocessing_and_feature_engineering.ipynb
├── phase_a_data_prep.ipynb
├── phase_b_training.ipynb
├── final_pipeline_with_arrays_appendix.ipynb
├── predict.py
├── test_smoke.py
├── sample_input.csv
├── sample_input_alarm.csv
├── expected_output.csv
├── expected_output_alarm.csv
├── Weather_Data.csv
├── Plant_Active_Power.csv
├── Array_Power.csv
├── phase_a/
├── phase_b/
└── model_artifacts/
```

All notebooks are kept at the project root or use the root-level `phase_a/` and `phase_b/` folders. Run them with the project root as the working directory so relative paths resolve correctly.

## Install dependencies

From the project root:

```bash
python -m pip install -r requirements.txt
```

## Run the notebooks

The supplied Phase-A and Phase-B outputs are already included. To reproduce the pipeline from raw inputs:

```text
1. Open preprocessing_and_feature_engineering.ipynb and run it from the project root.
2. Open phase_b_training.ipynb and run it from the project root.
3. Open anomaly_detection_final.ipynb and run it from the project root.
```

The final report loads the Phase-A and Phase-B artifacts, compares the saved forecasting results, computes GBM residuals, performs solar-elevation slice calibration, applies the four-row rolling score, and reports test alarms. It also writes diagnostic outputs under `results/` and `figures/` when executed.

`final_pipeline_with_arrays_appendix.ipynb` is the earlier deployment-pipeline notebook. `phase_a_data_prep.ipynb` is the Phase-A preparation notebook stored with the Phase-A outputs.

## Run in Google Colab

Upload `anomaly_detection_complete_project.zip`, then run this in the first cell:

```python
!unzip -q /content/anomaly_detection_complete_project.zip -d /content
%cd /content/anomaly_detection_complete_project
!pip install -q -r requirements.txt
```

Open `anomaly_detection_final.ipynb` from that folder and run all cells.

## Run the deployment smoke test

From the project root:

```bash
pytest -q test_smoke.py
```

The smoke test loads the sample input, checks the UTC index and output schema, compares predictions against `expected_output.csv`, and checks the alarm sample.

To run the deployment wrapper directly:

```bash
python predict.py
```

The deployment input must have a UTC-aware `DatetimeIndex` and these columns:

```text
irr_wm2, temp_c, mod_temp_c, plant_pu
```

The output includes `time_utc`, the input columns, `solar_elev_deg`, `yhat_pu`, `resid`, `slice`, `score_pt`, `score_sm4`, and `alarm`.

## Deployment artifacts

`model_artifacts/` contains the unchanged trained LightGBM model and calibration files:

- `model.txt`
- `scaler.json`
- `slices.json`
- `thresholds.json`
- `config.json`

The production threshold is stored in `thresholds.json` under `threshold_sm4`. Do not reorder features or change the slice calibration and threshold without recalibration.
