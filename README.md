# ☀️ Solar Plant Power Prediction & Fault Detection System

Samsung Innovation Campus (SIC8) · AI801 Graduation Project

<p align="center">
  <img src="https://img.shields.io/badge/PYTHON-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LIGHTGBM-185619?style=for-the-badge&logo=lightgbm&logoColor=white" />
  <img src="https://img.shields.io/badge/FASTAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/PANDAS-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/SCIKIT--LEARN-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

> *Can we see the invisible? An end-to-end MLOps and machine learning system to eliminate information blindness in solar plants through dual-system architecture: a real-time anomaly detector and a day-ahead power forecaster.*

---
> *Can we see the invisible? An end-to-end MLOps and machine learning system to eliminate information blindness in solar plants through dual-system architecture: a real-time anomaly detector and a day-ahead power forecaster.*

## 📌 Project Overview

Solar power plants often suffer from **Information Blindness**, which manifests in three critical ways:

1. **The Silent Revenue Bleed:** Array failures disguised as "cloudy days" leading to weeks of undetected generation loss.

2. **Alarm Fatigue:** Legacy threshold systems generating thousands of false alarms, causing operators to ignore real crises.

3. **Grid Penalties:** Financial penalties from grid operators due to inaccurate power output forecasting.

To solve these challenges, this system implements a **dual-system architecture**:

* **The Detector (Today):** Detects operational faults and underproduction without requiring historical labeled fault data.

* **The Forecaster (Tomorrow):** Predicts day-ahead power output (24–48 hours) using meteorological and historical data.

## 🗂️ Repository Structure

```
projects/
├── anomaly-detection/
│   ├── anomaly_detection_final.ipynb
│   ├── preprocessing_and_feature_engineering.ipynb
│   ├── phase_a/                    # Processed Phase-A inputs and metadata
│   ├── phase_b/                    # Phase-B training outputs
│   ├── model_artifacts/            # Trained deployment model and calibration
│   ├── predict.py                  # Production prediction wrapper
│   ├── test_smoke.py
│   ├── sample_input*.csv
│   ├── expected_output*.csv
│   ├── requirements.txt
│   └── README.md
└── forecasting/
    ├── forecasting_training_pipeline.ipynb
    ├── forecast_inputs/
    ├── artifacts/                  # Generated model outputs
    ├── f2_config.json
    ├── f3_config.json
    ├── f4_config.json
    ├── requirements.txt
    └── README.md
```

## ⚙️ System Architecture & Methodology

### 1. Data Processing & Cleaning

* **Dataset:** 8 years of data (2016–2023) across 14 solar arrays sampled at 15-minute resolution (96 rows/day).

* **Timezone Correction:** Rebuilt timestamps to UTC to prevent solar geometry errors and summer false alarms.

* **Robust Feature Engineering:** Derived solar elevation angles, cyclical time features (hour/day-of-year sine & cosine), and used Robust Scaling ($\frac{x - \text{median}}{\text{IQR}}$) to prevent fault distortion.

* **Leak-Free Splitting:** Strict time-based splits (Train: 2016–2021, Val: 2022, Test: 2023) with a 48-hour buffer to avoid lag leakage.

### 2. Anomaly Detection Pipeline

* **Expected Power Model:** Trained a Gradient Boosting / LightGBM model on clean data achieving an $R^2$ of 0.989 and low error (MAE = 0.02).

* **Slice Calibration:** Compensates for varying solar elevation throughout the day using 8 solar-elevation slices with independent medians and MADs.

* **Robust Alerting:** Uses a 1-hour rolling mean on z-scores against the 99.5th percentile validation threshold (4.616) to eliminate false alarms and keep the alarm rate under control (0.386% on test data).

### 3. Power Forecasting Pipeline

* Evaluated baselines, short-term models, and day-ahead models (LightGBM vs. Ridge vs. MLP).

* **LightGBM Winner:** Outperformed baselines by reducing RMSE by **2.6% in 2022** and **3.6% in early 2023**.

## 🚀 Quick Start & Installation

Create a virtual environment and install the dependencies for the desired pipeline:

### For Anomaly Detection:

```
cd projects/anomaly-detection
python -m venv .venv
.venv/bin/pip install -r requirements.txt
pytest -q test_smoke.py
```

### For Forecasting:

```
cd projects/forecasting
python -m venv .venv
.venv/bin/pip install -r requirements.txt
jupyter lab
```

## 📊 Key Results

* **Expected Power Model (R²):** 0.989 (GBM).

* **Test Alarm Rate:** 0.386% (47 alarm rows out of 12,180 scoreable rows on 2023 data).

* **Synthetic Fault Recall:**
  * **Drop Faults:** 81%
  * **Stuck Faults:** 100%

* **Forecasting Improvement:** Reduced error by up to **3.6%** compared to historical persistence baselines.

## 🌐 Web Application Deployment

The project includes an interactive web interface allowing operators to:

* Choose between normal samples and alarm-test samples, or upload custom CSV datasets.

* View actual vs. expected solar power output.

* Compute residuals and monitor real-time alarm triggers.

* Leverage **Explainable AI (XAI)** powered by LightGBM feature contributions to interpret individual model predictions.
