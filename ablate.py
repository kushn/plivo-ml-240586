"""Ablation: capacity x feature-set x calibration, on honest splits only."""
import importlib.util, os, sys, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("."))
from predict import read_labels, build_matrix, fit_calibrated, oof_predict
from features import FEATURE_GROUPS, group_index

spec = importlib.util.spec_from_file_location("scorer", "starter/score.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
TMP = "pred_tmp.csv"

DATA = {}
for lang in ["english", "hindi"]:
    d = f"eot_data/{lang}"
    df = read_labels(d)
    X = build_matrix(d, df, -1, quiet=True)
    y = (df.label == "eot").to_numpy().astype(int)
    DATA[lang] = (d, df, X, y)
print("features extracted:", {k: v[2].shape for k, v in DATA.items()}, flush=True)


def delay(d, df, p):
    pd.DataFrame({"turn_id": df.turn_id, "pause_index": df.pause_index,
                  "p_eot": p}).to_csv(TMP, index=False)
    r = S.score(d + "/labels.csv", TMP)
    return r["auc"], r["latency"] * 1000, r["cutoff"] * 100


def run(cols, method, **ov):
    out = {}
    for tr, te in [("english", "hindi"), ("hindi", "english")]:
        _, dtr, Xtr, ytr = DATA[tr]
        d_te, df_te, Xte, _ = DATA[te]
        m = fit_calibrated(Xtr[:, cols], ytr, dtr.turn_id.to_numpy(), method, **ov)
        out[f"{tr[:2]}->{te[:2]}"] = delay(d_te, df_te, m.predict_proba(Xte[:, cols])[:, 1])
    for lang in ["english", "hindi"]:
        d, df, X, y = DATA[lang]
        p = oof_predict(X[:, cols], y, df.turn_id.to_numpy(), method, **ov)
        out[f"oof-{lang[:2]}"] = delay(d, df, p)
    # pooled OOF across both languages = closest proxy to the hidden set
    Xa = np.vstack([DATA[l][2][:, cols] for l in ["english", "hindi"]])
    ya = np.concatenate([DATA[l][3] for l in ["english", "hindi"]])
    ga = np.concatenate([l[:2] + "_" + DATA[l][1].turn_id for l in ["english", "hindi"]])
    pa = oof_predict(Xa, ya, ga, method, **ov)
    from sklearn.metrics import roc_auc_score
    out["pooled_auc"] = roc_auc_score(ya, pa)
    return out


CAPACITY = {
    "old(400/15/15)": dict(max_iter=400, max_leaf_nodes=15, min_samples_leaf=15),
    "new(120/7/25)":  dict(max_iter=120, max_leaf_nodes=7, min_samples_leaf=25),
    "tiny(60/4/35)":  dict(max_iter=60, max_leaf_nodes=4, min_samples_leaf=35),
}

rows = []
for fname in ["all", "lite", "prosody"]:
    cols = group_index(fname)
    for cname, cap in CAPACITY.items():
        for method in ["isotonic", "sigmoid"]:
            r = run(cols, method, **cap)
            mean_delay = np.mean([r[k][1] for k in ["en->hi", "hi->en", "oof-en", "oof-hi"]])
            rows.append(dict(feats=f"{fname}({len(cols)})", cap=cname, calib=method,
                             pooled_auc=r["pooled_auc"], mean_delay=mean_delay,
                             **{k: r[k][1] for k in ["en->hi", "hi->en", "oof-en", "oof-hi"]}))
            print(f"{rows[-1]['feats']:12s} {cname:14s} {method:9s} "
                  f"auc={r['pooled_auc']:.3f} mean={mean_delay:6.0f}ms  "
                  + "  ".join(f"{k}={r[k][1]:.0f}" for k in ["en->hi","hi->en","oof-en","oof-hi"]),
                  flush=True)

res = pd.DataFrame(rows).sort_values("mean_delay")
res.to_csv("ablation_results.csv", index=False)
print("\n=== ranked ===")
print(res.to_string(index=False))
if os.path.exists(TMP):
    os.remove(TMP)
