# Method notes

## Layout

| Folder | What it does |
|---|---|
| `forecasting_engine/histories.py`, `store.py`, `eurostat.py` | local history store (parquet) and the loaders for ENTSO-E, Oil Bulletin, Spanish ports, Eurostat and the weather table |
| `forecasting_engine/lab/` | the forecasting lab: `panel.py` (frozen data per family, publication delays), `core.py` (origins, strict reference, models, scoring, ledger), `bench.py` (Chronos-2, TiRex-2, Toto 2.0, TimesFM 2.5, FlowState on the same cells), `blend_round.py` (blend of the champion with the best open variants), `scoreboard.py` (the comparison table), `chain.py` / `causal.py` (cross-layer inputs), `src_*.py` and `sources.py` (data loaders) |
| `forecasting_engine/cascade/` | cascade benchmark (`cascade.py`) and shock replays (`shocks.py`) |
| `forecasting_engine/whatif/` | five what-if mechanisms (`mech_*.py`, parameters in `data/*.json`), a small registry, and `backtests/` with the code that tested and calibrated each one |
| `forecasting_engine/gifteval/` | GIFT-Eval harness, the four zero-shot members, validation-only blend selection and scoring |
| `results/`, `data/` | small published outputs and the cascade event catalogue |

## Forecasting lab (rolling origin, locked holdout)

- A layer ("family") is a set of series (countries, ports, currencies) at one frequency: hourly, daily, weekly or monthly.
- Every input enters as it was known at the forecast origin: each driver has a publication delay (`panel.py`), and a value is used only after it was published.
- Each family's history is split once into development origins and a locked holdout (the last 20 %), written to `results/lab/registry.json`.
- Score: pooled skill = 1 - sum of scaled absolute errors / the same for the strict reference (the better of "no change" and "same as last season"), over all series, origins and steps.
- Candidates are tried on development only. A candidate gets one holdout look only if it beats the champion on the whole development window and on most of its four time blocks; it becomes champion only if it also wins on the holdout. Every run is logged with a code hash.
- Open models are scored on the same cells in their best fair variant (own history, covariates, cross-series). A layer counts as won only if the engine beats the best of Chronos-2, TiRex-2 and Toto 2.0.
- `results/lab/trials.jsonl` is the ledger with per-series detail removed; `bench.json` holds the open-model scores. `python -m forecasting_engine.lab.scoreboard` rebuilds `results/scoreboard.md` from them.

## Cascade benchmark

Cell = event x affected layer. Models forecast from origins that start before the shock onset through the event window, using only data published at each origin. Metrics: skill against the strict reference, direction, peak timing, size, which layers moved (against placebo windows), interval coverage. Models: the champion with all cross-layer inputs, the same champion with them removed (the ablation), Chronos-2 with the shock layer as covariates, Chronos-2, TiRex-2, Toto 2.0, and local projections. 21 development events and 8 locked check events (`data/cascade_events.json`). `python -m forecasting_engine.cascade.cascade aggregate --split dev|check` aggregates the shipped cell results.

## What-if engine

Given a driver path (a temperature path, a strike calendar, a policy-rate path, an oil path, a transit cut) each mechanism returns the response with p10 / p50 / p90, a status (validated, screening, experimental), the one-line back-test and the caveats. Mechanisms: heat -> electricity demand, strikes -> air traffic, rates -> borrowing cost, oil -> pump prices -> inflation, chokepoints -> freight. Parameters are small JSON files in `forecasting_engine/whatif/data/`; `whatif/backtests/build.py` rebuilds most of them from the back-test outputs in `results/cascade/`. Back-tests whose source data has restricted terms (ENTSO-E actual load, EUROCONTROL flights, Drewry freight indices) ship coefficients and ratios only, never the raw series.

## GIFT-Eval

`gifteval/forecast.py` runs the four members in their authors' notebook setups and writes quantile forecasts; `select_blend.py` scores every member subset and combiner on validation windows (before the test region) and picks one per domain x term group; `score.py` scores the test windows and aggregates as the leaderboard does (geometric mean of metric / Seasonal Naive). Needs the GIFT-Eval datasets under `$GIFT_ROOT` and the model weights under `$FE_MODEL_DIR`.

## Data and environment variables

No raw data is stored here. Loaders fetch public series (ENTSO-E, Eurostat, ECB, FRED, EIA, GDELT, Wikimedia, NASA POWER, ...) into `$FE_HISTORY_STORE` (default `data/history`).

| Variable | Meaning |
|---|---|
| `ENTSOE_TOKEN` | ENTSO-E Transparency API token (needed for the energy layers) |
| `FE_HISTORY_STORE` | history store folder (or `gs://...` with google-cloud-storage installed) |
| `FE_ALLOW_RESTRICTED` | `1` to use ENTSO-E items outside its free re-use list (back-tests only) |
| `FE_LAB_DIR` | folder with `registry.json`, `trials.jsonl`, `bench.json` (default `results/lab`) |
| `FE_FM_DEVICE`, `FE_MODEL_DIR`, `FE_FM_LOCAL` | device (`cpu` / `cuda`), local model folder, forecast cache for the foundation models |
| `FE_GRID` | parquet grid of Europe (land cover, population) used by the fire-weather inputs |
| `FE_WHATIF_OFF` | comma-separated what-if ids to switch off (default none) |
| `GIFT_ROOT`, `GIFT_PREDS`, `GIFT_OUT` | GIFT-Eval data, prediction files, output folder |

Not shipped, so some loaders need your own copy: the GDELT daily export (built from the public BigQuery dataset), the Europe grid, and the ERA5-type population-weighted temperature table (`weather/popw_daily.parquet`). The oil -> pump -> inflation parameter file is shipped as built; its builder depended on a bottom-up inflation model that is not included.
