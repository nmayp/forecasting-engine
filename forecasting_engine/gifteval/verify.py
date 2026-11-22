"""Compare member test scores with the authors' published rows. python -m forecasting_engine.gifteval.verify PREDS_DIR [configs...]"""
import os
import sys
import csv
from pathlib import Path

import numpy as np

from forecasting_engine.gifteval import data as gdata
from forecasting_engine.gifteval import harness as H

PUB = {"Chronos-2": "Chronos-2", "Toto-2.0": "Toto-2.0-1B", "TimesFM-2.5": "TimesFM-2.5", "TiRex-2": "TiRex-2-Zeroshot"}
PUBDIR = Path(os.environ.get("PUBDIR", "published"))      # the published GIFT-Eval result CSVs of each member


def pub(m):
    return {r["dataset"]: r for r in csv.DictReader(open(PUBDIR / f"{PUB[m]}.csv"))}


def main():
    pd_ = Path(sys.argv[1]); only = sys.argv[2:]
    P = {m: pub(m) for m in PUB}
    for ds, term in H.all_configs():
        name = H.config_name(ds, term)
        if only and name not in only:
            continue
        safe = name.replace("/", "__")
        have = [m for m in PUB if (pd_ / "test" / safe / f"{m}.npy").exists()]
        if not have:
            continue
        s = gdata.build_split(ds, term, "test", H.GIFT)
        ref = H.load_task(ds, term, H.GIFT)
        assert np.array_equal(np.nan_to_num(ref.targets, nan=-9e9), np.nan_to_num(s.tgts, nan=-9e9)), "order mismatch " + name
        t = gdata.to_task(s)
        for m in have:
            sc = H.score(t, np.load(pd_ / "test" / safe / f"{m}.npy"))
            r = P[m][name]
            pm, pc = float(r["eval_metrics/MASE[0.5]"]), float(r["eval_metrics/mean_weighted_sum_quantile_loss"])
            print(f"{name:34s} {m:12s} MASE {sc['MASE']:.4f} pub {pm:.4f} ({sc['MASE']/pm-1:+.1%})  CRPS {sc['CRPS']:.4f} pub {pc:.4f} ({sc['CRPS']/pc-1:+.1%})", flush=True)


if __name__ == "__main__":
    main()
