"""Rolling evaluation loop shared by all models, and metrics."""

from typing import Protocol

import numpy as np
import pandas as pd

from src.config import QUANTILE_COLUMNS

RESULT_COLUMNS = ["model", "target_date", "hour", "y_true", "y_pred", *QUANTILE_COLUMNS]


class Forecaster(Protocol):
    """Common interface. A model may keep an internal state as long as it only sees `history`."""

    def predict(self, history: pd.DataFrame, future_cov: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame | None]:
        """Return 24 point forecasts and, if available, the quantiles (columns q10 … q90)."""
        ...


def run_rolling(model: Forecaster, name: str, df: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Forecast each day of `days`, moving forward one day at a time; long-format results (RESULT_COLUMNS).

    Reads results/<name>.parquet if it exists instead of recomputing.
    """
    raise NotImplementedError


def mae(y_true, y_pred) -> float:
    raise NotImplementedError


def rmse(y_true, y_pred) -> float:
    raise NotImplementedError


def rmae(y_true, y_pred, y_naive) -> float:
    """Model MAE / reference naive MAE (D-7 on Monday, Saturday, Sunday; D-1 otherwise)."""
    raise NotImplementedError


def pinball(y_true, quantiles: pd.DataFrame) -> float:
    """Mean pinball loss over the 10th … 90th percentile levels."""
    raise NotImplementedError


def diebold_mariano(errors_a: pd.DataFrame, errors_b: pd.DataFrame, norm: int = 1) -> float:
    """One-sided multivariate Diebold-Mariano test (H1: model B is more accurate than model A).

    Errors are (days, 24) arrays; the loss differential is computed on the daily vector of
    24 errors. Reimplementation of `epftoolbox.evaluation.DM` (Apache-2.0, Lago et al. 2021).
    Returns the p-value.
    """
    raise NotImplementedError
