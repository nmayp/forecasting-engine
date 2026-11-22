# forecasting-engine

A forecasting lab for public data layers (energy, water, transport, economy, health, ...), with a cascade benchmark (multi-layer shocks) and a what-if engine
(strikes -> air traffic, heat waves -> power demand, rates -> borrowing costs, ...). It compares itself with Chronos-2, TiRex-2 and Toto 2.0 on a locked
holdout, and runs GIFT-Eval with a blend of those models.

Headline (25 Oct 2026): ahead of the best open model on 43 of 65 layers (58 of 65 against Chronos-2); GIFT-Eval MASE 0.675 / CRPS 0.462 on 97 datasets.
The tables are in `results/summary.md` and `results/scoreboard.md`. Notes on the method and the folders: `docs/methods.md`.

Per-layer champions, model pins and a data manifest to check the claims: `verification/`. What you need to supply yourself: `docs/inputs.md`.

Run the tests (no data, keys or GPU needed):

    pip install -r requirements.txt pytest
    python -m pytest tests -q

Rebuild the comparison table from the shipped ledger: `python -m forecasting_engine.lab.scoreboard`.
Try a what-if: `python -c "from forecasting_engine.whatif import registry as R; print(R.run('heat_load', R.EX.EXAMPLES['heat_load']['request'])['summary'])"`.

Data: nothing big is stored here. The loaders fetch public series (ENTSO-E, Eurostat, ECB, FRED, EIA, GDELT, ...) into a local folder (`FE_HISTORY_STORE`,
default `data/history`); the ENTSO-E loaders read their token from `ENTSOE_TOKEN`. `docs/methods.md` lists the variables and what has to be supplied
separately (GDELT export, Europe grid, weather table, GIFT-Eval data, model weights). Re-running the full lab needs those and a GPU.
