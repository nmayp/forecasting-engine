# Inputs

Most inputs are downloaded by the loaders from public sources. A few you have to supply yourself.

## Downloaded by the loaders

Eurostat, ECB, FRED, the EU Weekly Oil Bulletin, CISA KEV, the Oxford COVID-19 policy tracker, ECDC respiratory-virus data and the other public sources listed in `forecasting_engine/lab/sources.py` and `forecasting_engine/histories.py`. They write into the history store (`$FE_HISTORY_STORE`, default `data/history`). Run `python -m forecasting_engine.lab.sources all` for the ones in that module.

## Supply yourself

| Input | Where it goes | How to get it |
|---|---|---|
| ENTSO-E token | `ENTSOE_TOKEN` | free key from the ENTSO-E Transparency Platform |
| Daily temperature table (population-weighted mean, max, HDD, CDD per country) | `weather/popw_daily.parquet` in the history store | built from an ERA5-type reanalysis |
| GDELT 2.0 daily event counts | `gdelt/events_daily-*.parquet` in the history store | run the query from `sources.build_gdelt_sql()` in BigQuery and export it |
| Land-cover and population grid of Europe | `$FE_GRID` (default `data/cells_europe.parquet`) | only for the fire-weather inputs |
| GIFT-Eval datasets | `$GIFT_ROOT` | Salesforce's GIFT-Eval datasets from Hugging Face |
| Foundation-model weights | `$FE_MODEL_DIR`, or downloaded by id (see `verification/models.json`) | Hugging Face |

## Files written when you run a script

Some usage lines name a file under `results/`, for example `results/cascade/heat_load_holdout.json`. Those are written by that script and are not shipped.
