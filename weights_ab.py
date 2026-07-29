"""Does weighting training samples by hold duration buy latency?

The scorer only charges a false cutoff when the agent's action delay elapses
BEFORE the user resumes -- i.e. only on holds longer than the delay. So a false
positive on a 200 ms hold is free while one on a 1.5 s hold costs a turn.
pause_end is future information at inference and is never a feature, but it is
in the training labels, so it is legitimate to use it to WEIGHT training rows.
"""
import importlib.util, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("."))

from predict import read_labels, build_matrix, make_base
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location("scorer", "starter/score.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
TMP = "pred_tmp.csv"
CAP = dict(max_iter=120, max_leaf_nodes=7, min_samples_leaf=25)
SEED = 0

DATA = {}
for lang in ["english", "hindi"]:
    d = f"eot_data/{lang}"
    df = read_labels(d)
    DATA[lang] = (d, df, build_matrix(d, df, -1, quiet=True),
                  (df.label == "eot").to_numpy().astype(int),
                  (df.pause_end - df.pause_start).to_numpy())


def weights(y, dur, scheme):
    w = np.ones(len(y), dtype=float)
    if scheme == "none":
        return w
    hold = y == 0
    if scheme == "lin":          # longer hold -> costlier false positive
        w[hold] = 0.5 + np.clip(dur[hold], 0, 2.0)
    elif scheme == "step":       # only holds the agent could actually cut
        w[hold] = np.where(dur[hold] >= 0.85, 3.0, 0.4)
    elif scheme == "step_soft":
        w[hold] = np.where(dur[hold] >= 0.85, 2.0, 0.7)
    return w


def fit(X, y, g, w):
    n = max(2, min(5, pd.Series(g).nunique(), int(np.bincount(y).min())))
    cv = list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEED).split(X, y, g))
    m = CalibratedClassifierCV(make_base(**CAP), method="isotonic", cv=cv, n_jobs=1)
    m.fit(X, y, sample_weight=w)
    return m


def oof(X, y, g, w):
    p = np.zeros(len(y))
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y, g):
        p[te] = fit(X[tr], y[tr], np.asarray(g)[tr], w[tr]).predict_proba(X[te])[:, 1]
    return p


def delay(d, df, p):
    pd.DataFrame({"turn_id": df.turn_id, "pause_index": df.pause_index,
                  "p_eot": p}).to_csv(TMP, index=False)
    return S.score(d + "/labels.csv", TMP)["latency"] * 1000


for scheme in ["none", "lin", "step_soft", "step"]:
    res = {}
    for tr, te in [("english", "hindi"), ("hindi", "english")]:
        _, dtr, Xtr, ytr, dur_tr = DATA[tr]; d_te, df_te, Xte, _, _ = DATA[te]
        m = fit(Xtr, ytr, dtr.turn_id.to_numpy(), weights(ytr, dur_tr, scheme))
        res[f"{tr[:2]}->{te[:2]}"] = delay(d_te, df_te, m.predict_proba(Xte)[:, 1])
    for lang in ["english", "hindi"]:
        d, df, X, y, dur = DATA[lang]
        res[f"oof-{lang[:2]}"] = delay(d, df, oof(X, y, df.turn_id.to_numpy(), weights(y, dur, scheme)))
    Xa = np.vstack([DATA[l][2] for l in DATA]); ya = np.concatenate([DATA[l][3] for l in DATA])
    da = np.concatenate([DATA[l][4] for l in DATA])
    ga = np.concatenate([l[:2] + "_" + DATA[l][1].turn_id for l in DATA])
    pa = oof(Xa, ya, ga, weights(ya, da, scheme))
    print(f"{scheme:10s} pooled_auc={roc_auc_score(ya, pa):.4f} mean={np.mean(list(res.values())):6.0f}ms  "
          + "  ".join(f"{k}={v:.0f}" for k, v in res.items()), flush=True)
if os.path.exists(TMP):
    os.remove(TMP)
