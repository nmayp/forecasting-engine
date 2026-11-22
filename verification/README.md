# verification

What you need to check the layer-by-layer claims without the research tree. Rebuild it with `python -m forecasting_engine.lab.export_verification`.

| file | what it holds |
|---|---|
| `champions.json` | the champion of each of the 66 layers on the locked holdout: its recipe, its development and holdout skill, and how many holdout looks the layer used |
| `models.json` | the third-party models used as members and as the bar, by Hugging Face id and package version |
| `manifest.json` | per layer: series, development and holdout origins, strict reference, every driver with its publication delay, the public source of its shared inputs, and the SHA-256 of each series file as frozen at the time |

Things to know before relying on it:

- **Weights are rules.** A recipe such as `adapt:gbm_only-energy+weather_big|chronos2` is a mean weighted by recent errors; the weights are worked out at each forecast origin, so there is no weights file. The grammar is at the top of `champions.json`.
- **66 layers, 65 counted.** The scoreboard counts the layers where all three open models were scored.
- **Third-party weights are not copied.** For Chronos-2 the package version and weights revision were not recorded; for the other models the weights revision was not either (FlowState: `r1.1`).
- **A re-download will not match exactly.** Hashes tell you whether you have the same file; sources such as Eurostat revise their history.
- **Some sources cannot be shipped.** ENTSO-E layers need your own free token; the temperature table and the GDELT export are not included.
- **Not included yet:** a script that re-runs the holdout from these files.
