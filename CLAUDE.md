# Project: day-ahead electricity spot price forecasting (DE-LU)

Brief for an agent that generates the notebooks of this project. Read it in full before writing any code. All protocol decisions are made here: do not change them without flagging it.

## 1. Context

- Time series course project, Master of Science HES-SO (Lausanne).
- Team: Aladin and Léon.
- Final presentation around **24 November 2026**.
- Expectation: do something clean and draw conclusions from it. No strict grading rubric. Favour protocol rigour over sophistication.
- Hardware: development happens on two machines, kept in sync through GitHub: one without an NVIDIA GPU, one with an NVIDIA RTX 3070 (8 GB VRAM) that runs the foundation model evaluations. Code must pick the device at runtime (`cuda` if available, else `cpu`), never hard-code it. `data/` and `results/` are not versioned and must be copied between machines when needed. Compute time is not a hard constraint.
- Reuse the tools and vocabulary of the course (see section 9).

## 2. The business problem

On the European market, electricity is mainly traded the day before delivery through a **day-ahead** auction (European SDAC market coupling, gate closure around 12:00 Berlin time). It sets one price per period and per bidding zone for the next day.

The price is set by the last power plant dispatched in the **merit order**:
- lots of wind/solar (marginal cost ≈ 0) → low prices, sometimes **negative**;
- high demand with little renewable generation → gas plants → high prices.

Expected patterns in the series: daily seasonality (morning and evening peaks, midday solar dip), weekly seasonality (cheaper weekends), annual seasonality, plus weather-driven spikes and drops.

Chosen zone: **DE-LU (Germany-Luxembourg)**, highly sensitive to wind and solar, hence where covariates should help most.

## 3. Research question

> How much do covariates (load, wind and solar forecasts, calendar) improve day-ahead price forecasts, and do pre-trained foundation models used zero-shot (TimesFM 3, Timestral) beat the classical statistical models seen in the course?

The experimental design is a two-way table:

| | Without covariates | With covariates |
|---|---|---|
| **Classical (course)** | SARIMA | SARIMAX |
| **Foundation model** | TimesFM 3, Timestral | TimesFM 3 + cov., Timestral + cov. |

Reading: a row measures the contribution of covariates within a model family; a column compares families with the same information.

## 4. Task definition

### Notation

- **D** = target day (delivery day).
- The forecast is issued on day **D−1**, before auction gate closure (12:00 Berlin time).
- Target: the **24 hourly prices of day D**, in EUR/MWh, Berlin local time.
- **Rolling-origin evaluation**: repeat the forecast for every day D of the test period, moving forward one day at each iteration.

### Information available at the forecast origin (day D−1, before 12:00)

| Information | Usable? | Reason |
|---|---|---|
| Prices up to the end of day D−1 | Yes | D−1 prices are published on the afternoon of D−2 |
| Prices of day D | **No** | This is the target |
| Day-ahead load, wind and solar forecasts for D | Yes | Published by the transmission system operators (see limitation in section 10) |
| Past day-ahead forecasts (up to D−1) | Yes | History of forecasts already published |
| Calendar of D (hour, day of week, German public holidays) | Yes | Known with certainty |
| **Actual** load/wind/solar values | **No** | Never used, neither in the past nor in the future |

Absolute rule: no data posterior to the origin may enter a forecast, except the covariate forecasts for D and the calendar.

Covariates are **always forecasts**, including over the context window: a model must see the same kind of data in the past and in the future.

## 5. Data

### Single source: ENTSO-E Transparency Platform

Everything is downloaded with the Python package `entsoe-py` (API token available, read from the environment variable `ENTSOE_API_KEY`, never hard-coded). Zone code: DE-LU.

| Series | Nature | Unit |
|---|---|---|
| Day-ahead price | Actual auction result (target) | EUR/MWh |
| Day-ahead total load forecast | Forecast | MW → convert to GW |
| Day-ahead wind onshore + offshore forecast | Forecast, sum both into a single `wind_fc` series | MW → GW |
| Day-ahead solar forecast | Forecast | MW → GW |

- Check the exact method names and parameters in the `entsoe-py` documentation (do not guess).
- Use the **day-ahead process type** for wind/solar forecasts (not intraday).
- **Energy-Charts** (Fraunhofer ISE, https://api.energy-charts.info/, no registration) is used only as a spot check: compare a few days of prices to validate the ENTSO-E download.

### Calendar
- Hour of day (0–23), day of week, `non_working_day` flag (Saturday, Sunday or German national public holiday, `holidays` package).

### Periods
- **Download**: from 2023-01-01 to the last complete day available (avoids the 2021–2022 energy crisis regime).
- **Pre-test** (exploration, SARIMA order selection): 2023-01-01 → 2026-03-31.
- **Main test**: **2026-04-01 → 2026-09-30** (183 days). All models, same days.
- **Control period**: from 2026-10-16 to the last complete day available. It is posterior to the release of both foundation models and serves as a contamination test (section 7).

### Preprocessing
- **Resolution**: since 1 October 2025, the DE-LU day-ahead market uses 15-minute periods. Aggregate to hourly by **averaging the 4 quarter-hours**, in UTC, before converting to local time. Same treatment for the covariates.
- **Time zones**: store in UTC, then build the working table in Berlin time with **exactly 24 hours per day** (`epftoolbox` convention):
  - spring DST transition day (23 h): the missing hour is linearly interpolated between its neighbours;
  - autumn DST transition day (25 h): the two occurrences of the repeated hour are averaged.
  Document this convention in the notebook. Note: the main test period contains no DST transition (29 March and 25 October 2026); the control period contains one.
- **Missing values**: count and report them per series; linearly interpolate gaps of at most 3 h; beyond that, report and decide explicitly.
- **Negative prices and spikes**: real values, do not filter or clip them.
- Output of this step: a single file `data/dataset.parquet`, hourly index in local time, columns `price`, `load_fc`, `wind_fc`, `solar_fc`, `hour`, `dow`, `non_working_day`.
- Do not republish the raw data in a public repository; keep the download code for reproducibility.

## 6. Models

### Naive baselines
1. **Naive D−1**: price at the same hour the previous day.
2. **Naive D−7**: price at the same hour one week earlier.
3. **Reference naive** (`epftoolbox` convention): D−7 if D is a Monday, Saturday or Sunday, D−1 otherwise. **This is the rMAE reference.**

### Classical models (course)
4. **Holt-Winters** (ETS, lecture 5): **additive** seasonality (multiplicative is impossible with negative prices), period 168 h, no trend. Refitted every day on the last 8 weeks (1,344 h). Supplementary model: first to drop if problems arise.
5. **SARIMA** (lecture 4): season s = 24.
6. **SARIMAX** (lecture 4): same orders as SARIMA, plus the exogenous variables `load_fc`, `wind_fc`, `solar_fc`, `non_working_day`.

Rules for SARIMA/SARIMAX:
- **Order selection** in the exploration notebook, on the pre-test period only, following the course approach: seasonal differencing (1−B²⁴), ACF/PACF, then Ljung-Box test on the residuals. Suggested starting point: (1,0,1)(1,1,1)₂₄. The orders are then **frozen** for the whole test period.
- **Estimation** on a rolling window of the last 8 weeks, **refitted every Monday**. On other days, update the model state with the new observations without re-estimating the parameters (check the appropriate method in the `statsmodels` documentation).
- Use `statsmodels.tsa.statespace.sarimax.SARIMAX`, as in the course.

### Foundation models (zero-shot, no fine-tuning)
7. **TimesFM 3 univariate** and 8. **TimesFM 3 with covariates**.
9. **Timestral univariate** and 10. **Timestral with covariates**.

Common parameters:
- Context: **2,048 hours** (≈ 85 days) of prices for both models, horizon 24.
- With covariates: `load_fc`, `wind_fc`, `solar_fc` and the calendar, as past covariates (over the context) and known future covariates (over the 24 h of D).
- Retrieve the point forecast and the quantiles.

TimesFM 3:
- Google Research, released 31 August 2026. Code: GitHub repository `google-research/timesfm`. Weights: Hugging Face `google/timesfm-3.0-pytorch`.
- Non-commercial, non-production weight licence: fine for this academic project.
- Native support for past and known future covariates. Quantiles from the 10th to the 90th percentile.
- Package `timesfm` 3.0.2 on PyPI. The README examples use `TimesFM3Evaluator` from `timesfm3`, while the installed `timesfm` package exposes `TimesFM3Forecaster`: check which one is current before writing code.

Timestral:
- Mistral AI, released 15 October 2026. Sizes 5M, 28M, 123M, 209M: use **the largest (209M)**.
- Weights on Hugging Face (modified MIT licence), inference code on GitHub (Apache-2.0). Find Mistral's official repositories; do not guess the identifiers.
- Announced: past and known future covariates, context up to 32,768 steps, 21 quantiles (0.01, 0.05, 0.10 … 0.90 by 0.05, 0.95, 0.99, so the 10th … 90th percentiles are included). The announced performance has not been independently verified yet.
- Installed from git (`mistralai/timestral`, version 1.0.0). The model card gives no training data description or cutoff date.
- **`forecast()` clips forecasts to non-negative values by default (`clip_nonnegative=True`). Always pass `clip_nonnegative=False`: negative prices are real values.** This is the only deviation from default parameters.

For both: **do not guess the API**. Read the official README and examples (covariate format, checkpoint loading, outputs) before writing code. This information comes from a web search: confirm it in the official documentation.

## 7. Evaluation protocol

- Same test period, same days and same horizon for all models.
- For each test day and each model: 24 forecasts, stored in long format.
- **Metrics**:
  - **MAE** in EUR/MWh (main metric);
  - **rMAE** = model MAE / reference naive MAE (< 1: better than the naive);
  - RMSE;
  - mean **pinball loss** for the probabilistic models (TimesFM 3, Timestral), computed on the quantile levels common to both (10th … 90th percentile; check that Timestral provides these levels, otherwise interpolate and document it).
- **Diebold-Mariano test** between pairs of models (function available in `epftoolbox`), in particular: each model with covariates against its version without, and the best foundation model against SARIMAX.
- **Detailed analyses**:
  - error by hour of day;
  - working vs non-working days;
  - high vs low wind forecast days (e.g. quartiles of daily mean `wind_fc`);
  - behaviour on spikes and on negative-price hours;
  - covariate gain compared across families (SARIMA → SARIMAX vs TimesFM → TimesFM + cov. vs Timestral → Timestral + cov.).
- **Contamination test**: TimesFM 3 and Timestral were released during or after the main test period; their pre-training data might contain these prices. Read their model cards (cutoff date, training data) and compare the rMAE of all models on the control period. A foundation model clearly better on the main test than on the control period, while the other models are stable, is a contamination signal.

## 8. Code structure

```
data/                     # downloaded data (not versioned)
notebooks/
  01_download.ipynb       # ENTSO-E → preprocessing → data/dataset.parquet
  02_exploration.ipynb    # series, daily/weekly profiles, histograms, ACF/PACF, price vs covariates, SARIMA order selection
  03_baselines.ipynb      # naives, Holt-Winters
  04_sarima.ipynb         # SARIMA, SARIMAX, residual diagnostics
  05_foundation.ipynb     # TimesFM 3 and Timestral, univariate and with covariates
  06_results.ipynb        # tables, DM tests, detailed analyses, final plots
src/
  config.py               # protocol constants: periods, zone, context, horizon, columns
  data.py                 # dataset loading, context / future covariates split for a day D
  evaluation.py           # common rolling loop + metrics
results/                  # one parquet per model (not versioned)
figures/                  # final figures exported as PNG (versioned)
```

- **Common evaluation loop**: a function that, for each day D, builds `history` (all columns up to the end of D−1) and `future_cov` (covariates and calendar for the 24 h of D), calls `model.predict(history, future_cov)` and records the result. `predict` returns 24 point values and, if available, the quantiles.
- A model may keep an internal state (weekly SARIMA refit) as long as it only sees `history`.
- Results format: columns `model`, `target_date`, `hour` (0–23, local time), `y_true`, `y_pred`, `q10` … `q90` (empty if not available).
- Save forecasts to disk and do not recompute a model that has already been computed.
- Fix random seeds if a model is stochastic.
- Default parameters, no hyperparameter search.
- All code, comments, notebooks and documentation are written in English.

## 9. Link with the course

Use the tools seen in the course:

| Lecture | To use in the project |
|---|---|
| 1. Getting started | Naive, seasonal naive; out-of-sample evaluation; point forecast vs distribution (quantiles) |
| 2. Basics | Descriptive statistics, histograms, boxplots by hour/day, QQ plot of prices (heavy tails), trend/season/noise decomposition |
| 3. Correlations | ACF of prices (peaks expected at lags 24 and 168), stationarity, price vs covariates correlation (without concluding causality) |
| 4. ARIMA | Seasonal differencing, ACF/PACF for order selection, SARIMA/SARIMAX, residual diagnostics (`plot_diagnostics`, Ljung-Box) |
| 5. Smoothing | Additive Holt-Winters, additive vs multiplicative justification |

Possible opening in the conclusion: price volatility varies over time (calm and turbulent periods), which falls under the GARCH models covered later in the course.

## 10. Limitations to document

- **Covariate publication times**: the load forecast must be published before the auction, but the regulation allows wind/solar forecasts to be published until 18:00 the day before, hence potentially after gate closure. Common practice in the literature (`epftoolbox`), to be mentioned.
- Possible **contamination** of the foundation models (section 7).
- **Gas and CO₂ prices** are missing, although they often determine the marginal price.
- **Hourly aggregation**: the target is no longer exactly the traded product (quarter-hour).
- **DST convention** (section 5).

## 11. Final deliverable

A presentation (about 15–20 slides):
1. Context: day-ahead market, merit order, why forecast
2. Data: sources, preprocessing, exploration
3. Protocol: task, available information, rolling evaluation, metrics
4. Results: MAE/rMAE two-way table, forecast vs actual plots
5. Analysis: covariate contribution per family, failure cases, contamination
6. Conclusions and limitations

## 12. Schedule

| Dates | Work |
|---|---|
| 7–9 Nov. | Download and preprocessing (01), exploration and SARIMA order selection (02) |
| 10–16 Nov. | Evaluation loop, naives, Holt-Winters, SARIMA/SARIMAX (03, 04), first TimesFM 3 and Timestral calls, then full evaluation (05) |
| 17–23 Nov. | Results, DM tests, detailed analyses, contamination (06), slides |

## 13. What the agent must avoid

- Using actual load/wind/solar values, whether in the context or in the future.
- Using day D prices in the forecast of D.
- Choosing SARIMA orders or any other parameter by looking at the test period.
- Comparing models evaluated on different periods, days or horizons.
- Evaluating on in-sample fit instead of out-of-sample forecasts.
- Inventing API signatures (TimesFM, Timestral, `entsoe-py`, `statsmodels`, `epftoolbox`): check the documentation.
- Fine-tuning the foundation models or running a hyperparameter search.
- Hard-coding the ENTSO-E token.
