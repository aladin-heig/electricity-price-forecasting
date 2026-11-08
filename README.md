# Day-ahead electricity price forecasting (DE-LU)

Time series course project, Master of Science HES-SO (Lausanne), by Aladin and Léon.

## Question

How much do covariates (load, wind and solar forecasts, calendar) improve the forecast of next-day hourly prices in the Germany-Luxembourg bidding zone? And do foundation models used zero-shot beat classical statistical models?

| | Without covariates | With covariates |
|---|---|---|
| **Classical** | SARIMA | SARIMAX |
| **Foundation model** | TimesFM 3, Timestral | TimesFM 3 + cov., Timestral + cov. |

Baselines: naive D−1, naive D−7, `epftoolbox` reference naive, additive Holt-Winters.

## Protocol

- Forecast issued the day before auction gate closure, using only the information available at that time: past prices, published load, wind and solar forecasts, calendar. No actual covariate values.
- Rolling-origin evaluation, one day at a time, on the main test period (2026-04-01 → 2026-09-30, 183 days), same days for all models.
- Metrics: MAE, rMAE, RMSE, pinball loss, Diebold-Mariano test.
- Control period after the release of the foundation models (from 2026-10-16) to detect possible contamination of their training data.

The full protocol is described in [CLAUDE.md](CLAUDE.md).

## Results

Work in progress.

## Structure

```
notebooks/
  01_download.ipynb     # ENTSO-E → preprocessing → data/dataset.parquet
  02_exploration.ipynb  # exploratory analysis, SARIMA order selection
  03_baselines.ipynb    # naives, Holt-Winters
  04_sarima.ipynb       # SARIMA, SARIMAX
  05_foundation.ipynb   # TimesFM 3, Timestral
  06_results.ipynb      # tables, DM tests, detailed analyses
src/
  config.py             # protocol periods and parameters
  data.py               # loading, history / future covariates split
  evaluation.py         # common rolling loop, metrics
figures/                # final figures
```

## Reproduce

```bash
uv sync                      # install the environment
uv run nbstripout --install  # strip notebook outputs on commit
cp .env.example .env         # then set ENTSOE_API_KEY
```

Then run the notebooks in order. The data comes from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu) and is not redistributed in this repository: `01_download.ipynb` downloads it.

## License

Code under the MIT license. The data remains subject to ENTSO-E terms.
