"""Run the four members in their authors' GIFT-Eval setups. Usage:
python -m forecasting_engine.gifteval.forecast --split val|test|both --members A,B --configs a/b/c,... [--out preds2]"""
import argparse
import time
from pathlib import Path

import numpy as np

from forecasting_engine.gifteval import data as gdata
from forecasting_engine.gifteval import harness as H
from forecasting_engine.gifteval import members

ROOT = H.GIFT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="both")
    ap.add_argument("--members", default="Chronos-2,Toto-2.0,TimesFM-2.5,TiRex-2")
    ap.add_argument("--configs", default="")
    ap.add_argument("--out", default="preds")
    a = ap.parse_args()
    only = [c for c in a.configs.split(",") if c]
    splits = ["val", "test"] if a.split == "both" else [a.split]
    cfgs = [(ds, t) for ds, t in H.all_configs() if not only or H.config_name(ds, t) in only]
    sz = lambda c: sum(f.stat().st_size for f in (ROOT / c[0]).rglob("*.arrow"))
    cfgs.sort(key=lambda c: (sz(c), c[1]))
    import torch
    torch.set_float32_matmul_precision("high")      # TF32 matmuls (speed)
    log = open("forecast.log", "a")
    for split in splits:
        for ds, term in cfgs:
            name = H.config_name(ds, term); safe = name.replace("/", "__")
            todo = [m for m in a.members.split(",") if not (Path(a.out) / split / safe / f"{m}.npy").exists()]
            if not todo:
                continue
            s = gdata.build_split(ds, term, split, ROOT)
            if s.n_flat == 0:
                print("no data", name, split, flush=True); continue
            for m in todo:
                t0 = time.time()
                try:
                    p, nb = members.finalize(members.RUN[m](s), s)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"FAILED {m} {name} {split}: {e}", flush=True); continue
                d = Path(a.out) / split / safe; d.mkdir(parents=True, exist_ok=True)
                np.save(d / f"{m}.npy", p)
                el = time.time() - t0
                msg = f"{name},{split},{m},{el:.1f},n={s.n_flat},nonfinite={nb}"
                print(msg, flush=True); log.write(msg + "\n"); log.flush()
            del s


if __name__ == "__main__":
    main()
