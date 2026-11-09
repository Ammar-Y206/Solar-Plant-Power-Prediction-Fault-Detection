from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "model_artifacts"
FEATURES = [
    "irr_wm2",
    "temp_c",
    "mod_temp_c",
    "solar_elev_deg",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]
REQUIRED_COLUMNS = ["irr_wm2", "temp_c", "mod_temp_c", "plant_pu"]
OUTPUT_COLUMNS = [
    "time_utc",
    "irr_wm2",
    "temp_c",
    "mod_temp_c",
    "plant_pu",
    "solar_elev_deg",
    "yhat_pu",
    "resid",
    "slice",
    "score_pt",
    "score_sm4",
    "alarm",
]

with (ARTIFACT_DIR / "scaler.json").open() as f:
    SCALER = json.load(f)
with (ARTIFACT_DIR / "slices.json").open() as f:
    SLICES = json.load(f)
with (ARTIFACT_DIR / "thresholds.json").open() as f:
    THRESHOLDS = json.load(f)
with (ARTIFACT_DIR / "config.json").open() as f:
    CONFIG = json.load(f)
MODEL = lgb.Booster(model_file=str(ARTIFACT_DIR / "model.txt"))

PARAMS = SCALER["params"]
INNER_EDGES = np.asarray(SLICES["inner_edges"], dtype=float)
MED_S = np.asarray(SLICES["med_s"], dtype=float)
SIG_S = np.asarray(SLICES["sig_s"], dtype=float)
LATITUDE = float(CONFIG.get("latitude", 40.5))
LONGITUDE = float(CONFIG.get("longitude", 21.8))


def solar_elevation(t_utc, lat=LATITUDE, lon=LONGITUDE):
    """Return solar elevation in degrees using the NOAA approximation."""
    times = pd.DatetimeIndex(t_utc)
    n = (times - pd.Timestamp("2000-01-01 12:00", tz="UTC")) / pd.Timedelta(days=1)
    n = np.asarray(n, dtype=float)
    L = np.deg2rad((280.460 + 0.9856474 * n) % 360)
    g = np.deg2rad((357.528 + 0.9856003 * n) % 360)
    lam = L + np.deg2rad(1.915) * np.sin(g) + np.deg2rad(0.020) * np.sin(2 * g)
    eps = np.deg2rad(23.439 - 0.0000004 * n)
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    lst = np.deg2rad(((280.46061837 + 360.98564736629 * n) + lon) % 360)
    ha = lst - ra
    la = np.deg2rad(lat)
    return np.rad2deg(
        np.arcsin(
            np.sin(la) * np.sin(dec)
            + np.cos(la) * np.cos(dec) * np.cos(ha)
        )
    )


def _features(df):
    idx = df.index
    features = df[["irr_wm2", "temp_c", "mod_temp_c"]].copy()
    features["solar_elev_deg"] = solar_elevation(idx)
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    doy = idx.day_of_year.to_numpy()
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    features["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return features[FEATURES]


def _scale_features(features):
    med = np.asarray([PARAMS[c]["median"] for c in FEATURES], dtype=float)
    scale = np.asarray([PARAMS[c]["scale"] for c in FEATURES], dtype=float)
    return ((features[FEATURES].to_numpy(dtype=float) - med) / scale).astype(np.float32)


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Score observations and return the deployment output contract.

    Input rows are scored chronologically for the rolling score. The returned
    rows retain the caller's original order.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a UTC tz-aware DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("df must have a UTC tz-aware DatetimeIndex; naive index is not allowed")

    work = df.copy()
    work.index = work.index.tz_convert("UTC")
    work["__input_order"] = np.arange(len(work), dtype=np.int64)
    ordered = work.sort_index(kind="mergesort")

    features = _features(ordered)
    yhat_scaled = MODEL.predict(_scale_features(features))
    yhat_pu = yhat_scaled * PARAMS["plant_pu"]["scale"] + PARAMS["plant_pu"]["median"]

    out = ordered[["irr_wm2", "temp_c", "mod_temp_c", "plant_pu"]].copy()
    out["solar_elev_deg"] = features["solar_elev_deg"]
    out["yhat_pu"] = yhat_pu
    out["resid"] = out["plant_pu"] - out["yhat_pu"]
    out["slice"] = np.searchsorted(INNER_EDGES, out["solar_elev_deg"].to_numpy(), side="right")
    out["score_pt"] = np.abs(
        (out["resid"].to_numpy() - MED_S[out["slice"].to_numpy()])
        / SIG_S[out["slice"].to_numpy()]
    )
    out["score_sm4"] = out["score_pt"].rolling(4, min_periods=1).mean()
    out["alarm"] = out["score_sm4"] >= float(THRESHOLDS["threshold_sm4"])
    out["__input_order"] = ordered["__input_order"].to_numpy()
    out = out.sort_values("__input_order", kind="mergesort")
    out["time_utc"] = out.index
    return out[OUTPUT_COLUMNS].reset_index(drop=True)


if __name__ == "__main__":
    sample = pd.read_csv(BASE_DIR / "sample_input.csv", index_col="time_utc", parse_dates=True)
    print(predict(sample).to_string(index=False))
