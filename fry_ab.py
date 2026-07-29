"""A/B the vocal-fry mask in isolation (single-process so we can toggle it)."""
import importlib.util, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("."))

import features as F
from predict import read_labels, build_matrix, fit_calibrated, oof_predict
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location("scorer", "starter/score.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
TMP = "pred_tmp.csv"
CAP = dict(max_iter=120, max_leaf_nodes=7, min_samples_leaf=25)


def harvest():
    out = {}
    for lang in ["english", "hindi"]:
        d = f"eot_data/{lang}"
        df = read_labels(d)
        X = build_matrix(d, df, 1, quiet=True)          # n_jobs=1 -> in-process
        out[lang] = (d, df, X, (df.label == "eot").to_numpy().astype(int))
    return out


def delay(d, df, p):
    pd.DataFrame({"turn_id": df.turn_id, "pause_index": df.pause_index,
                  "p_eot": p}).to_csv(TMP, index=False)
    r = S.score(d + "/labels.csv", TMP)
    return r["latency"] * 1000


def evaluate(DATA, tag):
    res = {}
    for tr, te in [("english", "hindi"), ("hindi", "english")]:
        _, dtr, Xtr, ytr = DATA[tr]; d_te, df_te, Xte, _ = DATA[te]
        m = fit_calibrated(Xtr, ytr, dtr.turn_id.to_numpy(), "isotonic", **CAP)
        res[f"{tr[:2]}->{te[:2]}"] = delay(d_te, df_te, m.predict_proba(Xte)[:, 1])
    for lang in ["english", "hindi"]:
        d, df, X, y = DATA[lang]
        res[f"oof-{lang[:2]}"] = delay(d, df, oof_predict(X, y, df.turn_id.to_numpy(), "isotonic", **CAP))
    Xa = np.vstack([DATA[l][2] for l in ["english", "hindi"]])
    ya = np.concatenate([DATA[l][3] for l in ["english", "hindi"]])
    ga = np.concatenate([l[:2] + "_" + DATA[l][1].turn_id for l in ["english", "hindi"]])
    auc = roc_auc_score(ya, oof_predict(Xa, ya, ga, "isotonic", **CAP))
    print(f"{tag:22s} pooled_auc={auc:.4f} mean={np.mean(list(res.values())):6.0f}ms  "
          + "  ".join(f"{k}={v:.0f}" for k, v in res.items()), flush=True)
    return auc, np.mean(list(res.values()))


# --- fry mask ON (as written) --------------------------------------------- #
on = evaluate(harvest(), "fry-mask ON")

# --- fry mask OFF: threshold so deep nothing is ever flagged --------------- #
F.FRY_DROP_DB = 1e9
off = evaluate(harvest(), "fry-mask OFF")

print(f"\ndelta (ON - OFF): auc {on[0]-off[0]:+.4f}   mean delay {on[1]-off[1]:+.0f} ms")

# --- what does the slope look like now? ------------------------------------ #
F.FRY_DROP_DB = 10.0
D = harvest()
i_slope = F.FEATURE_NAMES.index("f0_slope_st_per_s")
i_fry = F.FEATURE_NAMES.index("f0_fry_frac")
for lang in ["english", "hindi"]:
    d, df, X, y = D[lang]
    s, fr = X[:, i_slope], X[:, i_fry]
    print(f"{lang}: median f0_slope  eot={np.nanmedian(s[y==1]):+.2f}  hold={np.nanmedian(s[y==0]):+.2f} st/s"
          f" | mean fry_frac eot={np.nanmean(fr[y==1]):.3f} hold={np.nanmean(fr[y==0]):.3f}")
if os.path.exists(TMP):
    os.remove(TMP)
