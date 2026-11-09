# Forecasting Training Pipeline

This package contains one Jupyter Notebook with the actual training workflow for F2, F3, and F4. It starts from the raw split files in `forecast_inputs/`, builds the train-only maps and causal features, trains the models, evaluates fresh in-memory predictions, and writes all outputs to `artifacts/`.

## Run

From this directory:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab
```

Open `forecasting_training_pipeline.ipynb` and run all cells. The notebook uses CPU-compatible LightGBM, scikit-learn, and PyTorch. The PyTorch MLP uses an explicit Adam optimizer, MSE loss, minibatch training, validation early stopping, and seeded initialization.

The notebook is also suitable for non-interactive execution:

```bash
jupyter nbconvert --to notebook --execute forecasting_training_pipeline.ipynb \
  --output forecasting_training_pipeline_executed.ipynb
```

Run it from the package directory so the relative paths resolve to `forecast_inputs/` and `artifacts/`.

## Inputs

`forecast_inputs/` contains the compressed train, validation, and test target/quality/weather tables plus the events and preprocessing metadata used by the original protocol. The three JSON files in the package root are the F2, F3, and F4 configuration records.

## Outputs

The notebook creates `artifacts/` with:

- `f2_metrics.csv` and `f2_predictions.csv.gz` for climatology and persistence baselines.
- `f3_tuning.csv`, `f3_metrics.csv`, `f3_predictions.csv.gz`, and `f3_feature_importance.csv`.
- `f3_models/` containing one LightGBM text model for every F3 horizon, feature set, and seed.
- `f4_tuning.csv`, `f4_summary.csv`, `f4_predictions.csv.gz`, and `f4_feature_importance.csv`.
- `f4_models/` containing LightGBM text models, Ridge joblib files, and PyTorch MLP state dictionaries.
- `f4_imputer.joblib`, `f4_scaler.joblib`, `train_only_maps.json`, and `run_config.json`.

Metrics are calculated from predictions produced in the same notebook run. The notebook does not read pre-computed prediction CSVs for evaluation.

## Protocol notes

- Environment and climatology maps are fit from the cleaned training split only.
- F3 uses horizons 4, 8, 16, and 24 fifteen-minute steps, PWR and PWR+WX features, LightGBM early stopping, a validation hyperparameter search, and seeds 42, 43, and 44.
- F4 uses daily UTC origins with horizons 97–192, daylight targets with `env_t >= 0.05`, corrected target-time `same_time_*` features, LightGBM, Ridge, and a PyTorch MLP.
- March 2022 is excluded from the primary validation comparison but remains available in the raw split for stress testing.
- Nighttime is outside the learned F4 daylight target scope; the notebook evaluates the valid daylight rows used by the protocol.
