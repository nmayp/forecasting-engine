"""The four members, each reproducing its authors' GIFT-Eval notebook (SalesforceAIResearch/gift-eval/notebooks).

Chronos-2   chronos-2.ipynb : native multivariate (variates jointly), predict_batches_jointly=True (cross learning),
            batch_size=100, each rolling window predicted separately, model default context (8192), float32.
Toto-2.0    toto_2_0.ipynb  : Toto2GluonTSModel, context_length 4096 (patch-aligned, minus horizon patches), native
            multivariate + past_feat_dynamic_real as extra variates, ffill imputation with observed mask,
            has_missing_values=True, scaler_fallback_min_obs=8, quantile_real_cap_k=1e4, decode_block_size=None.
TimesFM-2.5 timesfm2p5.ipynb: univariate (variates split), raw context incl. NaN, batches of 1024, max_context
            min(15360, longest context rounded up to 32), max_horizon 1024, flip invariance, continuous quantile head.
TiRex-2     tirex-2.ipynb   : univariate (variates split), full context, past covariates if present, checkpoint
            default TTA settings, forecast_gluon batch_size 512 (internal long-horizon handling).
All return (n_flat, 9, pl) float32, position order of the Split.
"""
import math
import os
from pathlib import Path
import numpy as np

try:
    import torch
except ImportError:  # the harness and the blend scoring run without torch
    torch = None

MODELS_DIR = Path(os.environ.get("FE_MODEL_DIR", "models"))
DEV = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
_M = {}
_PCB = {}


def _flat_items(s):
    """univariate view: list over flat positions of (unit_idx, variate)"""
    items = [None] * s.n_flat
    for i in range(len(s.units)):
        for v in range(s.V):
            items[s.positions[i, v]] = (i, v)
    return items


def _oom(e):
    return (torch is not None and isinstance(e, torch.cuda.OutOfMemoryError)) or "out of memory" in str(e).lower()


def get_model(name):
    if name in _M:
        return _M[name]
    if name == "Toto-2.0":
        from toto2 import Toto2Model
        try:   # forward-identity multiply (fwd_scale == 1.0) in unit-scaling's scale_bwd: pure overhead, same values
            import dd_unit_scaling.scale as _sc
            _orig = _sc._ScaledGrad.apply
            _sc._ScaledGrad.apply = staticmethod(lambda X, f, b: X if (f == 1.0 and not X.requires_grad) else _orig(X, f, b))
        except Exception as e:
            print("scale patch skipped", e)
        m = Toto2Model.from_pretrained(MODELS_DIR / "Datadog__Toto-2.0-1B", map_location=DEV).to(DEV).eval()
    elif name == "Chronos-2":
        from chronos import BaseChronosPipeline
        m = BaseChronosPipeline.from_pretrained(MODELS_DIR / "amazon__chronos-2", device_map=DEV, torch_dtype=torch.float32)
    elif name == "TiRex-2":
        from tirex2 import load_model
        m = load_model(MODELS_DIR / "NX-AI__TiRex-2-gifteval-zs", device=DEV)
    elif name == "TimesFM-2.5":
        import timesfm
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODELS_DIR / "google__timesfm-2.5-200m-pytorch", torch_compile=False)
        if DEV == "cuda":
            m.model.to("cuda")
    else:
        raise ValueError(name)
    _M[name] = m
    return m


# ------------------------------------------------------------------ Chronos-2
def run_chronos2(s, bs=100):
    m = get_model("Chronos-2")
    out = np.zeros((s.n_flat, 9, s.pl), dtype=np.float32)
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for k in range(s.windows):
        idx = [i for i, u in enumerate(s.units) if u["k"] == k]
        if not idx:
            continue
        inputs = []
        for i in idx:
            c = s.units[i]["tgt"][:, :s.units[i]["cut"]]
            inputs.append({"target": c[0] if s.V == 1 else c})
        b = bs
        while True:
            try:
                q, _ = m.predict_quantiles(inputs=inputs, prediction_length=s.pl, batch_size=b,
                                           quantile_levels=levels, predict_batches_jointly=True)
                break
            except Exception as e:
                if _oom(e) and b > 1:
                    torch.cuda.empty_cache(); b //= 2; print("chronos OOM ->", b, flush=True)
                else:
                    raise
        for i, qi in zip(idx, q):                      # qi: (V, pl, 9)
            arr = qi.float().cpu().numpy().transpose(0, 2, 1)
            out[s.positions[i]] = arr
    return out


# ------------------------------------------------------------------ Toto-2.0
def run_toto(s, bs=64, ctx_len=4096):
    m = get_model("Toto-2.0")
    patch = m.config.patch_size
    c_adj = ctx_len - (math.ceil(s.pl / patch) - 1) * patch
    past_len = max(patch, (c_adj // patch) * patch)
    out = np.zeros((s.n_flat, 9, s.pl), dtype=np.float32)
    order = sorted(range(len(s.units)), key=lambda i: -min(s.units[i]["cut"], past_len))
    nv = s.V + (s.units[0]["pf"].shape[0] if s.units[0]["pf"] is not None else 0)
    # memory budget: bs * nv sequences
    b = max(1, int(bs * 4 // max(nv, 1))) if nv > 1 else bs
    pos = 0
    while pos < len(order):
        chunk = order[pos:pos + b]
        L = max(min(s.units[i]["cut"], past_len) for i in chunk)
        L = ((L + patch - 1) // patch) * patch
        X = np.zeros((len(chunk), nv, L), np.float32)
        M = np.zeros((len(chunk), nv, L), bool)
        for j, i in enumerate(chunk):
            u = s.units[i]
            c = u["cut"]; n = min(c, past_len)
            arr = u["tgt"][:, c - n:c]
            if u["pf"] is not None:
                arr = np.concatenate([arr, u["pf"][:, c - n:c]], 0)
            obs = ~np.isnan(arr)
            filled = arr.copy()
            # ffill, leading NaN -> 0 (toto2 ffill_imputation)
            for r in range(filled.shape[0]):
                row = filled[r]
                nanm = np.isnan(row)
                if nanm.any():
                    idx = np.where(~nanm, np.arange(n), -1)
                    np.maximum.accumulate(idx, out=idx)
                    row = np.where(idx >= 0, row[np.maximum(idx, 0)], 0.0)
                    filled[r] = row
            X[j, :, L - n:] = filled
            M[j, :, L - n:] = obs
        try:
            Xt = torch.tensor(X, device=DEV); Mt = torch.tensor(M, device=DEV)
            sid = torch.zeros((len(chunk), nv), dtype=torch.long, device=DEV)
            with torch.no_grad():
                q = m.forecast({"target": Xt, "target_mask": Mt, "series_ids": sid}, s.pl, decode_block_size=None,
                               has_missing_values=True, scaler_fallback_min_obs=8, quantile_real_cap_k=1e4)
        except Exception as e:
            if _oom(e) and b > 1:
                torch.cuda.empty_cache(); b = max(1, b // 2); print("toto OOM ->", b, flush=True); continue
            raise
        q = q.float().cpu().numpy()                    # (9, b, nv, pl)
        for j, i in enumerate(chunk):
            out[s.positions[i]] = q[:, j, :s.V, :].transpose(1, 0, 2)
        pos += len(chunk)
    return out


# ------------------------------------------------------------------ TimesFM-2.5
def run_timesfm(s, batch=1024):
    import timesfm
    m = get_model("TimesFM-2.5")
    p = getattr(m.model, "p", 32)
    out = np.zeros((s.n_flat, 9, s.pl), dtype=np.float32)
    for b0 in range(0, s.n_flat, batch):
        ctx = []
        for c in s.ctxs[b0:b0 + batch]:
            c = np.asarray(c)
            if len(c) == 0 or np.isnan(c).all():
                c = np.zeros(max(1, len(c)), dtype=np.float32)
            ctx.append(c)
        mx = ((max(len(c) for c in ctx) + p - 1) // p) * p
        pcb = _PCB.get('v', 128)
        while True:
            try:
                m.compile(timesfm.ForecastConfig(max_context=min(15360, mx), max_horizon=1024, infer_is_positive=True,
                                                 use_continuous_quantile_head=True, fix_quantile_crossing=True,
                                                 force_flip_invariance=True, return_backcast=False,
                                                 normalize_inputs=True, per_core_batch_size=pcb))
                _, full = m.forecast(horizon=s.pl, inputs=ctx)
                break
            except Exception as e:
                if _oom(e) and pcb > 1:
                    torch.cuda.empty_cache(); pcb //= 2; _PCB['v'] = pcb; print("timesfm OOM ->", pcb, flush=True)
                else:
                    raise
        out[b0:b0 + len(ctx)] = np.asarray(full)[:len(ctx), :s.pl, 1:10].transpose(0, 2, 1)
    return out


# ------------------------------------------------------------------ TiRex-2
def run_tirex2(s, batch=512):
    from tirex2 import TimeseriesType
    m = get_model("TiRex-2")
    items = _flat_items(s)
    out = np.zeros((s.n_flat, 9, s.pl), dtype=np.float32)
    # sort by length so batches pad little (series are independent in TiRex-2 univariate mode)
    order = sorted(range(s.n_flat), key=lambda p: -len(s.ctxs[p]))
    has_pf = any(u['pf'] is not None for u in s.units)
    step = 128 if has_pf else 256
    pos = 0
    while pos < len(order):
        chunk = order[pos:pos + step]
        ts = []
        for p in chunk:
            i, v = items[p]
            u = s.units[i]
            t = torch.tensor(np.asarray(s.ctxs[p], dtype=np.float32)).reshape(1, -1)
            pc = None if u["pf"] is None else torch.tensor(u["pf"][:, :u["cut"]])
            ts.append(TimeseriesType(target=t, past_covariates=pc, future_covariates=None))
        try:
            fc = m.forecast(ts, prediction_length=s.pl, output_type="numpy", batch_size=step)
        except Exception as e:
            if _oom(e) and step > 1:
                torch.cuda.empty_cache(); step = max(1, step // 2); print("tirex OOM ->", step, flush=True); continue
            raise
        for p, f in zip(chunk, fc):
            out[p] = np.asarray(f)[0]
        pos += len(chunk)
    return out


RUN = {"Chronos-2": run_chronos2, "Toto-2.0": run_toto, "TimesFM-2.5": run_timesfm, "TiRex-2": run_tirex2}


def finalize(preds, s):
    """replace non-finite with last observed value (should be ~never), enforce monotone quantiles"""
    bad = ~np.isfinite(preds)
    nb = int(bad.sum())
    if nb:
        for p in np.where(bad.any(axis=(1, 2)))[0]:
            c = s.ctxs[p]; c = c[~np.isnan(c)]
            preds[p][bad[p]] = c[-1] if len(c) else 0.0
    return np.sort(preds, axis=1), nb
