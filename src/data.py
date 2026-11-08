"""Dataset loading and split of the information available for a target day D."""

import pandas as pd

from src.config import DATASET_PATH


def load_dataset(path=DATASET_PATH) -> pd.DataFrame:
    """Load data/dataset.parquet (hourly index in Berlin time, 24 h per day).

    Columns: price, load_fc, wind_fc, solar_fc, hour, dow, non_working_day.
    """
    raise NotImplementedError


def split_for_day(df: pd.DataFrame, day: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the information available at the forecast origin for target day `day`.

    Returns:
    - history: all columns up to the end of D-1 inclusive;
    - future_cov: covariates (forecasts) and calendar for the 24 h of D, without the price column.
    """
    raise NotImplementedError
