# RUNLOG

Metric: mean response delay (ms) at <= 5% interrupted turns, via `starter/score.py`.
Baseline to beat: silence-only, scored **per folder** - 1600 ms on english but
only 850 ms on hindi (see Run 6). The handout's single ~1600 ms reference does
not hold for both.

**Evaluation protocol.** Every number below is from held-out turns. Four honest
cells are reported: `en->hi` / `hi->en` (train one language, test the other) and
`oof-en` / `oof-hi` (grouped 5-fold out-of-fold within a language, turns never
split across folds). `mean` is their average. In-sample numbers are recorded only
where they are the point of the entry.

---

## Run 1 - first end-to-end pipeline

59 causal features (F0 slope in semitones, RMS decay, final-run lengthening,
trailing MFCC mean/std/delta, turn position). HistGradientBoosting,
`max_iter=400, max_leaf_nodes=15, min_samples_leaf=15`, isotonic calibration.

| split | AUC | delay |
|---|---|---|
| in-sample (english) | 1.000 | 100 ms |
| in-sample (hindi) | 1.000 | 100 ms |
| en->hi | 0.644 | 850 ms |
| hi->en | 0.612 | 1393 ms |
| oof-en | 0.603 | 1310 ms |
| oof-hi | 0.709 | 850 ms |

**The 100 ms is not a result.** The model was trained on both folders and then
scored on those same folders; 400 boosting iterations over 496 samples separate
the training set perfectly. Real performance is the 0.60-0.71 AUC band,
mean delay ~1090 ms. Recorded here because the gap between in-sample 1.000 and
OOF 0.695 is exactly what motivated Run 2.

---

## Run 2 - ablation: capacity x feature set x calibration

18 configurations, each scored on all four honest splits.
Full grid in `ablation_results.csv`.

| feature set | capacity | calib | pooled AUC | mean delay |
|---|---|---|---|---|
| all (60) | 120/7/25 | isotonic | **0.705** | **1078 ms** |
| all (60) | 400/15/15 | isotonic | 0.697 | 1079 ms |
| all (60) | 400/15/15 | sigmoid | 0.707 | 1114 ms |
| lite (38) | 120/7/25 | isotonic | 0.685 | 1105 ms |
| prosody (28) | 120/7/25 | isotonic | 0.650 | 1115 ms |
| prosody (28) | 400/15/15 | sigmoid | 0.638 | 1138 ms |

Three findings, two of which contradicted the hypotheses that motivated the run:

1. **Capacity reduction: no real gain, kept anyway.** Cutting to
   `max_iter=120, max_leaf_nodes=7, min_samples_leaf=25` moved held-out AUC
   0.697 -> 0.705 and mean delay 1079 -> 1078 ms, i.e. nothing. It did close the
   in-sample/OOF gap, and it trains ~3x faster, so it is kept for honesty and
   speed, not for score. **The overfitting was never costing held-out points -
   it was only making the in-sample numbers meaningless.**

2. **Feature pruning HURT. Hypothesis rejected.** The premise was that 39 MFCC
   columns on 496 samples were noise diluting the prosody. They are not: AUC
   falls monotonically as they are removed (all 0.705 -> lite 0.685 -> prosody
   0.650), and mean delay rises. Kept all 60. See Run 3 for why.

3. **Calibration: isotonic over sigmoid.** Sigmoid won marginally on AUC (0.707
   vs 0.705) but isotonic won on the metric that counts (1078 vs 1103 ms).
   Ranking and calibration are not the same objective here - the scorer sweeps
   one threshold across all turns, so what matters is that a given probability
   means the same thing in every turn, which is what lets the optimizer pick a
   lower threshold without breaching the 5% cutoff budget.

---

## Run 3 - vocal-fry masking, and why MFCCs were load-bearing

Added a creak gate to the YIN tracker: within the final 200 ms, voiced frames
sitting >= 10 dB below the window's voiced median are excluded from the F0
regression (turn-final creak keeps yin reporting garbage, often octave-halved,
pitch at collapsed glottal amplitude). Also exposed the masked fraction as a
feature, `f0_fry_frac`.

| | pooled AUC | mean delay |
|---|---|---|
| fry mask ON | 0.7046 | 1078 ms |
| fry mask OFF | 0.6997 | 1078 ms |

Small, real, kept (+0.005 AUC, delay unchanged). But the diagnostic printed
alongside it was the actual finding:

| | eot | hold |
|---|---|---|
| english median `f0_slope` | **+1.97 st/s** | -3.76 st/s |
| hindi median `f0_slope` | **-2.60 st/s** | +0.22 st/s |

**The terminal pitch cue is discriminative in both languages but with opposite
sign.** Hindi matches the textbook prior (turn-final fall); English in this
corpus does the reverse. That single fact explains the whole shape of the
results table: it is why `hi->en` (1393 ms) is the worst cell, and it is why
pruning MFCCs hurt - the MFCC block is acting as an implicit language/channel
identifier that lets the trees condition the sign of the prosodic rule. Remove
it and the model is forced into one global sign that is wrong half the time.

---

## Run 4 - duration-aware sample weighting (rejected)

The scorer only charges a false cutoff when the action delay elapses *before*
the user resumes, so a false positive on a 200 ms hold is free while one on a
1.5 s hold costs a turn. `pause_end` is future information at inference and is
never a feature, but it is in the training labels, so it is legitimate as a
training *weight*. Tested three schemes:

| weighting | pooled AUC | mean delay |
|---|---|---|
| none | 0.7046 | 1078 ms |
| linear in hold duration | 0.7082 | 1087 ms |
| step, 3.0 if dur >= 0.85 s else 0.4 | 0.7014 | 1071 ms |
| step_soft, 2.0 / 0.7 | 0.6904 | 1082 ms |

Best and worst differ by 16 ms - one grid step of the scorer's delay sweep, on
100 turns. **Rejected as noise, not signal.** Reverting rather than banking a
number I cannot defend in the discussion.

---

## Run 5 - extended feature set (28 new columns)

Added multi-scale slopes (250 ms / 1 s / 2 s for both F0 and RMS), voice-quality
features (spectral tilt, jitter, shimmer, HNR proxy), run-duration statistics,
and a turn-level signature (mean MFCC + F0 median/IQR + speech rate over ALL
audio so far) intended as an explicit channel/language variable to replace the
one the trees were inferring from trailing cepstra. 60 -> 88 features,
39 ms/pause.

| model | pooled AUC | mean delay |
|---|---|---|
| HGB, 60 features (run 2 winner) | 0.705 | 1078 ms |
| HGB, 88 features | 0.709 | 1111 ms |
| ExtraTrees, 88 features | 0.697 | 1064 ms |
| HGB+ET+LR ensemble | 0.708 | 1082 ms |

**The extra features did essentially nothing** (+0.004 AUC, delay slightly
worse). The explicit language signature did not beat the implicit one. Kept
because ExtraTrees on top of them is the best cell and the columns are cheap,
but this was not the win. Ensembling did not beat its best member either.

---

## Run 6 - scoring the given baseline per folder (the important one)

Ran `starter/baseline.py` + `score.py` on each folder, which no earlier run had
done:

| folder | silence-only baseline | our model (run 5) |
|---|---|---|
| english | 1600 ms | 1210 ms |
| hindi | **850 ms** | 850 ms |

**The handout's ~1600 ms reference does not hold per folder.** Under 5% of hindi
turns contain a hold longer than 850 ms, so an always-fire policy with an 850 ms
timer already meets the budget without any model. Every previous entry that
described hindi as "-47% vs baseline" was comparing against the wrong number:
on hindi we tie the timer, we do not beat it. English is the only folder where
the model has been earning anything.

Diagnostic behind it - achievable delay per action delay `d`, hindi:

| d | turns with a hold > d | AUC(eot vs those holds) | best delay |
|---|---|---|---|
| 0.50 s | 20 | 0.628 | 1248 ms |
| 0.70 s | 12 | 0.679 | 898 ms |
| 0.85 s | 5 | 0.675 | 850 ms |

The whole hindi score reduces to separating true ends from ~12 long holds. That
is the target, not overall AUC.

---

## Run 7 - cost-aware negatives + probability floor (shipped)

Two changes from the run 6 diagnostic:

1. **Weight training rows by what a mistake costs.** A false positive on a
   200 ms hesitation is free (the delay never elapses); one on a 1 s hold costs a
   turn. Short holds are ~80% of negatives, so equal weighting spends capacity on
   a distinction worth nothing. Weights: eot 1.0, hold >= 0.5 s 3.0, shorter
   hold 0.15. `pause_end` stays out of the features - it is used only as a
   training weight, which is label information, not future audio.
2. **Floor the emitted probabilities at 0.05**, the scorer's lowest swept
   threshold. Monotone, so no ranking changes, but it keeps the always-fire
   operating point reachable. Without it a model can score *worse* than doing
   nothing - measured at 857 ms on hindi against the 850 ms timer.

| model x scheme | mean delay | oof-en |
|---|---|---|
| ET, flat weights | 1064 ms | 1210 ms |
| **ET, cost-aware (shipped)** | **1047 ms** | **1134 ms** |
| ET, drop-short-holds | 1069 ms | 1190 ms |
| HGB, cost-aware | 1105 ms | 1216 ms |

Shipped run, regenerated end to end:

| folder | baseline | ours (grouped OOF) | operating point |
|---|---|---|---|
| english | 1600 ms | **1130 ms** (-29%) | threshold 0.50, delay 600 ms |
| hindi | 850 ms | 850 ms (tie) | threshold 0.05, delay 850 ms |

---

## Current shipped configuration

88 causal features; ExtraTrees (500 trees, `min_samples_leaf=4`,
`max_features="sqrt"`, balanced) under isotonic calibration over 5 grouped
folds; cost-aware sample weights; predictions floored at 0.05.

| deliverable | AUC | delay | baseline |
|---|---|---|---|
| `predictions_en.csv` (grouped OOF) | 0.620 | **1130 ms** | 1600 ms |
| `predictions_hi.csv` (grouped OOF) | 0.706 | **850 ms** | 850 ms |
| `predictions.csv` | both folders concatenated; scores identically | | |

`eot_model.joblib` is trained on english+hindi pooled and is what `predict.py`
loads for an unseen folder. The shipped per-folder CSVs are grouped out-of-fold
(`--oof`), because the provided folders are in the model's training set.

Expect the hidden set (unseen turns, mostly Hindi) to land near the hindi
column. Whether that beats its own baseline depends on that folder's hold-length
distribution, which sets where the timer alone lands - the honest claim is
"english -29%, hindi at parity", not a single headline number.

---

## Next, in priority order

1. **Rank turn ends above long holds specifically.** The metric reduces to ~12-20
   pauses per folder; a pairwise/ranking objective restricted to (eot, hold>0.5s)
   pairs targets it directly, where log-loss does not.
2. **Listen to those specific long holds.** They are few enough to hear in ten
   minutes, and no run so far has used a single listening pass.
3. Explicit language conditioning still unexplored as a *routing* choice
   (per-language experts + an audio language classifier), as opposed to the
   turn-signature features tried in run 5, which failed.
4. Multi-scale windows and voice-quality features are in but unhelpful so far -
   worth revisiting only after the objective change in (1).
