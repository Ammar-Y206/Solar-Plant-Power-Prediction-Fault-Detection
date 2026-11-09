from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import OUTPUT_COLUMNS, predict

BASE_DIR = Path(__file__).resolve().parent


def test_clean_sample_matches_expected():
    sample = pd.read_csv(
        BASE_DIR / "sample_input.csv",
        index_col="time_utc",
        parse_dates=True,
    )
    expected = pd.read_csv(BASE_DIR / "expected_output.csv")
    assert isinstance(sample.index, pd.DatetimeIndex)
    assert str(sample.index.tz) == "UTC"
    assert sample.index.is_monotonic_increasing

    actual = predict(sample)
    assert actual.columns.tolist() == OUTPUT_COLUMNS
    assert actual.shape == expected.shape
    actual_times = pd.to_datetime(actual["time_utc"], utc=True)
    expected_times = pd.to_datetime(expected["time_utc"], utc=True)
    assert actual_times.equals(expected_times)

    for column in ["yhat_pu", "resid", "score_pt", "score_sm4"]:
        np.testing.assert_allclose(actual[column], expected[column], rtol=0, atol=1e-9)
    assert actual["alarm"].tolist() == expected["alarm"].tolist()


def test_alarm_sample_has_alarm():
    sample = pd.read_csv(
        BASE_DIR / "sample_input_alarm.csv",
        index_col="time_utc",
        parse_dates=True,
    )
    expected = pd.read_csv(BASE_DIR / "expected_output_alarm.csv")
    actual = predict(sample)
    assert actual["alarm"].any()
    assert actual["alarm"].tolist() == expected["alarm"].tolist()


def test_naive_index_is_rejected():
    sample = pd.read_csv(BASE_DIR / "sample_input.csv", index_col="time_utc")
    with np.testing.assert_raises(ValueError):
        predict(sample)
