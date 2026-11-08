"""Protocol constants, shared by all notebooks.

Single source of truth for the periods and parameters described in CLAUDE.md:
do not redefine them locally in a notebook.
"""

from pathlib import Path

import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
DATASET_PATH = DATA_DIR / "dataset.parquet"

# Market
ZONE = "DE_LU"  # entsoe-py zone code, to be confirmed in its documentation
TZ = "Europe/Berlin"

# Periods (calendar days in Berlin time, bounds inclusive)
DOWNLOAD_START = pd.Timestamp("2023-01-01")
PRETEST_START = pd.Timestamp("2023-01-01")
PRETEST_END = pd.Timestamp("2026-03-31")
TEST_START = pd.Timestamp("2026-04-01")
TEST_END = pd.Timestamp("2026-09-30")
CONTROL_START = pd.Timestamp("2026-10-16")
CONTROL_END = pd.Timestamp("2026-11-08")  # last complete day of the download (01_download.ipynb)

TEST_DAYS = pd.date_range(TEST_START, TEST_END, freq="D")
assert len(TEST_DAYS) == 183

# Dataset columns
TARGET = "price"
COVARIATES = ["load_fc", "wind_fc", "solar_fc"]
CALENDAR = ["hour", "dow", "non_working_day"]
# Exogenous variables for SARIMAX and for the foundation models with covariates (same information)
EXOG = ["load_fc", "wind_fc", "solar_fc", "non_working_day"]

# Task
HORIZON = 24  # 24 hourly prices of day D

# Classical models
FIT_WINDOW_HOURS = 8 * 7 * 24  # 8 weeks = 1,344 h (SARIMA, SARIMAX, Holt-Winters)
SARIMA_SEASON = 24
HW_SEASON = 168

# Foundation models
FM_CONTEXT_LENGTH = 2048
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QUANTILE_COLUMNS = [f"q{int(round(q * 100))}" for q in QUANTILES]

# Preprocessing
MAX_INTERP_GAP_HOURS = 3

SEED = 42
