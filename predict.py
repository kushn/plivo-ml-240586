"""End-of-turn detector: CLI required by the assignment.

    python predict.py --data_dir eot_data/english --out predictions.csv

Behaviour
---------
* If a saved model (`eot_model.joblib` next to this script, or --model) exists,
  it is loaded and applied to `--data_dir`. This is the "unseen folder" path:
  no labels are needed beyond the schema, and nothing is refit.
* Otherwise the script trains on `--data_dir` itself and writes *out-of-fold*
  probabilities (grouped by turn_id, so no turn appears in its own training
  fold), then saves the full-data model for reuse. This keeps the emitted
  predictions honest instead of near-perfect in-sample fits.

Train explicitly on both languages and save one model:

    python predict.py --train --train_dir eot_data/english eot_data/hindi \
        --model eot_model.joblib

Model: HistGradientBoostingClassifier wrapped in CalibratedClassifierCV
(isotonic) over grouped folds. The scorer sweeps a probability threshold, so
calibrated, well-spread scores matter as much as raw AUC.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from features import FEATURE_GROUPS, FEATURE_NAMES, extract_features, group_index, load_audio

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "eot_model.joblib")
N_SPLITS = 5
SEED = 0


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def read_labels(data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    df["pause_index"] = df["pause_index"].astype(int)
    df["pause_start"] = df["pause_start"].astype(float)
    df = df.sort_values(["turn_id", "pause_index"], kind="stable").reset_index(drop=True)
    # previous pause start within the same turn -- past information only
    df["prev_pause_start"] = df.groupby("turn_id")["pause_start"].shift(1)
    return df


def _features_for_turn(data_dir: str, audio_file: str, rows) -> list[np.ndarray]:
    """One wav load per turn; all of its pauses extracted from that waveform."""
    x, sr = load_audio(os.path.join(data_dir, audio_file))
    out = []
    for ps, pi, prev in rows:
        out.append(
            extract_features(
                x, sr, ps, pi, None if prev is None or np.isnan(prev) else float(prev)
            )
        )
    return out


def build_matrix(data_dir: str, df: pd.DataFrame, n_jobs: int = -1, quiet=False):
    groups = df.groupby("audio_file", sort=False)
    order, jobs = [], []
    for audio_file, g in groups:
        order.extend(g.index.tolist())
        jobs.append(
            delayed(_features_for_turn)(
                data_dir,
                audio_file,
                list(zip(g["pause_start"], g["pause_index"], g["prev_pause_start"])),
            )
        )
    t0 = time.time()
    if not quiet:
        print(f"extracting features for {len(df)} pauses "
              f"across {len(jobs)} turns ...", flush=True)
    chunks = Parallel(n_jobs=n_jobs, backend="loky")(jobs)
    # `chunks` follows grouped order; map it back onto label-file row order
    inv = np.empty(len(order), dtype=int)
    inv[np.asarray(order)] = np.arange(len(order))
    X = np.vstack([np.vstack(c) for c in chunks])[inv]
    if not quiet:
        print(f"  done in {time.time() - t0:.1f}s -> X{X.shape}", flush=True)
    return X


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
# Capacity deliberately small: ~500 training pauses. The first version
# (max_iter=400, max_leaf_nodes=15) hit AUC 1.000 in-sample against 0.695
# out-of-fold -- pure memorisation. See RUNLOG.md.
HGB_PARAMS = dict(
    max_iter=120,
    learning_rate=0.06,
    max_leaf_nodes=7,
    min_samples_leaf=25,
    l2_regularization=1.0,
    early_stopping=False,
    class_weight="balanced",
)
MODEL = "et"                 # ExtraTrees > HGB on the metric; see RUNLOG run 6
ET_PARAMS = dict(n_estimators=500, min_samples_leaf=4, max_features="sqrt",
                 class_weight="balanced")
CALIBRATION = "isotonic"     # beat sigmoid on mean delay in the ablation
FEATURE_SET = "all"          # pruning the MFCC block HURT transfer; see RUNLOG


P_FLOOR = 0.05      # == the scorer's lowest swept threshold; see below


def make_base(**overrides):
    """ExtraTrees beat HGB on the metric (1047 vs 1077 ms mean); see RUNLOG."""
    if MODEL == "hgb":
        return HistGradientBoostingClassifier(random_state=SEED,
                                              **{**HGB_PARAMS, **overrides})
    return ExtraTreesClassifier(random_state=SEED, n_jobs=-1,
                                **{**ET_PARAMS, **overrides})


def cost_weights(y, dur):
    """Weight training rows by what a mistake on them actually costs.

    The scorer charges a false cutoff only when the action delay elapses BEFORE
    the user resumes, so firing on a 200 ms hesitation is free while firing on a
    1 s hold costs a turn. Short holds are ~80% of the negatives; left at equal
    weight they spend the model's capacity on a distinction worth nothing.

    `pause_end` is future information at inference and is never a feature - it
    is used here only to weight TRAINING rows, which is label information, not
    a causality violation.
    """
    return np.where(y == 1, 1.0, np.where(dur >= 0.5, 3.0, 0.15))


def apply_floor(p):
    """Guarantee we can never score worse than the silence-only timer.

    The scorer sweeps thresholds from 0.05 up. If any pause scores below 0.05 it
    can never fire, so a model CAN come out behind the trivial always-fire
    policy (measured: 857 ms vs the 850 ms baseline on hindi). Compressing into
    [0.05, 1.0] is monotone - it changes no ranking - but keeps the always-fire
    operating point reachable, making the baseline a floor rather than a coin
    flip.
    """
    return P_FLOOR + (1.0 - P_FLOOR) * np.asarray(p, dtype=float)


def fit_calibrated(X, y, groups, method=None, w=None, **overrides):
    """HGB + probability calibration on grouped folds (turns never split).

    The scorer sweeps a single threshold across all turns, so the ranking is
    only half the job: the probabilities must mean the same thing everywhere
    for a lower threshold to buy latency without breaking the 5% cutoff
    budget. Calibration is fit on held-out folds, never in-sample.
    """
    n_splits = min(N_SPLITS, int(pd.Series(groups).nunique()), int(np.bincount(y).min()))
    n_splits = max(2, n_splits)
    cv = list(
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        .split(X, y, groups)
    )
    clf = CalibratedClassifierCV(
        make_base(**overrides), method=method or CALIBRATION, cv=cv, n_jobs=1
    )
    clf.fit(X, y, sample_weight=w)
    return clf


def oof_predict(X, y, groups, method=None, w=None, **overrides):
    """Out-of-fold calibrated probabilities for the rows we trained on."""
    n_splits = max(2, min(N_SPLITS, int(pd.Series(groups).nunique())))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    p = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y, groups):
        m = fit_calibrated(X[tr], y[tr], np.asarray(groups)[tr], method,
                           None if w is None else w[tr], **overrides)
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def write_predictions(df: pd.DataFrame, p: np.ndarray, out: str):
    pd.DataFrame(
        {"turn_id": df["turn_id"], "pause_index": df["pause_index"],
         "p_eot": np.round(apply_floor(p), 6)}
    ).to_csv(out, index=False)
    print(f"wrote {len(df)} predictions -> {out}")


def main():
    ap = argparse.ArgumentParser(description="EOT probability per pause.")
    ap.add_argument("--data_dir", help="folder with labels.csv + audio/")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train", action="store_true",
                    help="fit a model and save it to --model")
    ap.add_argument("--train_dir", nargs="+", default=None,
                    help="folders to train on (default: --data_dir)")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--features", default=FEATURE_SET, choices=sorted(FEATURE_GROUPS),
                    help="feature subset (default: all)")
    ap.add_argument("--oof", action="store_true",
                    help="ignore any saved model: self-train and emit honest "
                         "out-of-fold probabilities for --data_dir")
    args = ap.parse_args()
    cols = group_index(args.features)

    # ---------------- training mode ------------------------------------- #
    if args.train:
        dirs = args.train_dir or ([args.data_dir] if args.data_dir else None)
        if not dirs:
            sys.exit("--train needs --train_dir or --data_dir")
        Xs, ys, gs, ws = [], [], [], []
        for d in dirs:
            df = read_labels(d)
            Xs.append(build_matrix(d, df, args.n_jobs)[:, cols])
            yy = (df["label"] == "eot").to_numpy().astype(int)
            ys.append(yy)
            ws.append(cost_weights(yy, (df["pause_end"] - df["pause_start"]).to_numpy()))
            gs.append((os.path.basename(os.path.abspath(d)) + "/" + df["turn_id"]).to_numpy())
        X, y, g = np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)
        w = np.concatenate(ws)
        p = oof_predict(X, y, g, w=w)
        print(f"grouped OOF AUC = {roc_auc_score(y, p):.4f}  "
              f"(n={len(y)}, eot rate={y.mean():.3f})")
        model = fit_calibrated(X, y, g, w=w)
        dump({"model": model, "feature_names": FEATURE_NAMES,
              "features": args.features}, args.model)
        print(f"saved model -> {args.model}")
        if args.data_dir:
            df = read_labels(args.data_dir)
            Xd = build_matrix(args.data_dir, df, args.n_jobs)[:, cols]
            print("WARNING: --data_dir was part of --train_dir; these "
                  "predictions are IN-SAMPLE. Use --oof for an honest score.")
            write_predictions(df, model.predict_proba(Xd)[:, 1], args.out)
        return

    if not args.data_dir:
        sys.exit("--data_dir is required")

    df = read_labels(args.data_dir)
    X_full = build_matrix(args.data_dir, df, args.n_jobs)

    # ---------------- inference on an unseen folder ---------------------- #
    if os.path.exists(args.model) and not args.oof:
        bundle = load(args.model)
        if bundle.get("feature_names") != FEATURE_NAMES:
            sys.exit("feature set changed since the model was saved; retrain "
                     f"with:  python predict.py --train --train_dir {args.data_dir}")
        p = bundle["model"].predict_proba(X_full[:, group_index(bundle.get("features", "all"))])[:, 1]
        print(f"loaded {args.model}")
        write_predictions(df, p, args.out)
        return

    X = X_full[:, cols]

    # ---------------- self-train, emit honest out-of-fold ---------------- #
    if "label" not in df.columns:
        sys.exit(f"no model at {args.model} and no labels to train on")
    y = (df["label"] == "eot").to_numpy().astype(int)
    g = df["turn_id"].to_numpy()
    print(f"training on {args.data_dir}; emitting grouped out-of-fold scores")
    w = cost_weights(y, (df["pause_end"] - df["pause_start"]).to_numpy())
    p = oof_predict(X, y, g, w=w)
    print(f"grouped OOF AUC = {roc_auc_score(y, p):.4f}  "
          f"(n={len(y)}, eot rate={y.mean():.3f})")
    if not args.oof:      # --oof is an evaluation mode; don't clobber the model
        model = fit_calibrated(X, y, g, w=w)
        dump({"model": model, "feature_names": FEATURE_NAMES,
              "features": args.features}, args.model)
        print(f"saved model -> {args.model}")
    write_predictions(df, p, args.out)


if __name__ == "__main__":
    main()
