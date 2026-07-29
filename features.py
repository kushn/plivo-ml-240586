"""Causal feature extraction for end-of-turn (EOT) detection.

HARD CAUSALITY RULE
-------------------
For a pause annotated at `pause_start`, every feature here is computed from
`x[: int(pause_start * sr)]` and nothing else. The very first thing
`extract_features` does is truncate the waveform; no downstream helper ever
sees a sample from at or after `pause_start`, so `pause_end` / pause duration
(future information for a `hold`) can never leak in.

Feature families
----------------
1. Prosodic pitch  : F0 contour over the last 500 ms, slope in semitones/s,
                     final-vs-local-reference offset, range, voiced fraction.
2. Energy decay    : RMS(dB) trend into the pause, short/long slopes, drop
                     relative to the loudness of the turn so far.
3. Timing / rhythm : final voiced-run length (final-syllable lengthening),
                     syllable-rate proxy, turn position, pause_index.
4. Spectral        : trailing MFCC mean/std (+ delta mean) over the last
                     500 ms -- captures the segmental identity of whatever
                     the speaker trailed off on.

Requires: numpy, librosa (which pulls in scipy/soundfile).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import medfilt

import librosa

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
SR = 16_000                 # data is 16 kHz mono; we resample if it is not
HOP = 160                   # 10 ms hop
WIN = 400                   # 25 ms analysis window for RMS
SHORT_S = 0.50              # "trailing" window: last 500 ms before the pause
LONG_S = 1.50               # prosodic context window
REF_S = 3.00                # local speaker-reference window for pitch/energy
N_MFCC = 13
F0_MIN, F0_MAX = 60.0, 400.0
F0_FRAME = 1024             # 64 ms; yin needs >= 2 periods at fmin (533 samples)
FRY_TAIL_S = 0.20           # window searched for turn-final creak
FRY_DROP_DB = 10.0          # dB below the voiced median that counts as fry

_EPS = 1e-10


def _names() -> list[str]:
    n = [
        # --- pitch -------------------------------------------------------- #
        "f0_slope_st_per_s",      # semitones/sec over last 500 ms (falling < 0)
        "f0_slope_long",          # same regression over last 1.5 s
        "f0_last_minus_med_st",   # final F0 vs local (3 s) speaker reference
        "f0_range_st",            # voiced range in the trailing window
        "f0_std_st",
        "f0_voiced_frac",         # voicing density right before the pause
        "f0_final_st",            # final F0, semitones re 100 Hz
        "f0_delta_last_two",      # last - previous voiced frame (micro-trend)
        "f0_fry_frac",            # share of trailing voiced frames that creak
        # --- energy ------------------------------------------------------- #
        "rms_slope_db_per_s",     # decay rate over last 500 ms
        "rms_slope_long",         # decay rate over last 1.5 s
        "rms_last_db",
        "rms_last_minus_p90",     # how far below the turn's speech level
        "rms_drop_300ms",         # dB fallen across the final 300 ms
        "rms_std_db",
        "rms_frac_below_20db",    # fraction of trailing frames near-silent
        # --- timing / rhythm ---------------------------------------------- #
        "final_run_s",            # duration of the last voiced/loud run
        "final_run_ratio",        # that run vs the mean run in the window
        "run_rate_hz",            # syllable-rate proxy (runs per second)
        "voiced_frac_long",
        "elapsed_s",              # pause_start (time talked so far)
        "log_elapsed",
        "pause_index",
        "since_prev_pause_s",     # gap since the previous annotated pause
        # --- spectral ----------------------------------------------------- #
        "zcr_mean",
        "spec_centroid_last",
        "spec_rolloff_last",
        "spec_flatness_last",
    ]
    n += [f"mfcc{i}_mean" for i in range(N_MFCC)]
    n += [f"mfcc{i}_std" for i in range(N_MFCC)]
    n += [f"dmfcc{i}_mean" for i in range(6)]
    # --- multi-scale: the 500 ms window was a guess, never tuned ----------- #
    n += ["f0_slope_250ms", "f0_slope_1s", "f0_slope_2s",
          "rms_slope_250ms", "rms_slope_1s", "rms_slope_2s",
          "energy_ratio_250_500"]
    # --- voice quality: phrase-final position is marked by breathy/creaky
    #     phonation in both languages ------------------------------------- #
    n += ["spec_tilt_db", "jitter_pct", "shimmer_db", "hnr_proxy_db",
          "voiced_run_final_s", "run_dur_mean", "run_dur_std", "run_dur_max"]
    # --- turn-level signature: a channel/speaker/language fingerprint from
    #     ALL audio so far. Gives the model an explicit conditioning variable
    #     for the sign-flipped prosody rule instead of making it infer one
    #     from the trailing cepstra. -------------------------------------- #
    n += [f"turn_mfcc{i}" for i in range(8)]
    n += ["turn_f0_med_st", "turn_f0_iqr_st", "turn_rms_p90", "turn_voiced_frac",
          "turn_run_rate_hz"]
    return n


FEATURE_NAMES: list[str] = _names()
N_FEATURES: int = len(FEATURE_NAMES)

# Named subsets, so the model can be trained on fewer columns without
# re-extracting anything. `prosody` is the transfer-safe core; the MFCC block
# is the part most likely to encode language/channel identity.
_PROSODY = [n for n in FEATURE_NAMES if not n.startswith(("mfcc", "dmfcc"))]
FEATURE_GROUPS: dict[str, list[str]] = {
    "all": list(FEATURE_NAMES),
    "prosody": _PROSODY,
    "lite": _PROSODY + [f"mfcc{i}_{s}" for i in range(5) for s in ("mean", "std")],
}


def group_index(name: str) -> np.ndarray:
    """Column indices of a named feature subset."""
    keep = set(FEATURE_GROUPS[name])
    return np.array([i for i, n in enumerate(FEATURE_NAMES) if n in keep], dtype=int)


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def load_audio(path: str, sr: int = SR):
    """Mono float32 waveform at `sr`."""
    x, _sr = librosa.load(path, sr=sr, mono=True)
    return x.astype(np.float32), _sr


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _tail(x: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    n = int(seconds * sr)
    return x[-n:] if len(x) > n else x


def _slope(values: np.ndarray, hop_s: float) -> float:
    """Least-squares slope of `values` per second. NaN-safe, 0.0 if degenerate."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return 0.0
    t = np.arange(len(v), dtype=float)[ok] * hop_s
    v = v[ok]
    if np.ptp(t) < 1e-6:
        return 0.0
    return float(np.polyfit(t, v, 1)[0])


def _hz_to_st(f0: np.ndarray) -> np.ndarray:
    """Hz -> semitones re 100 Hz. Pitch is perceived (and declines) log-linearly."""
    f0 = np.asarray(f0, dtype=float)
    out = np.full(f0.shape, np.nan)
    ok = np.isfinite(f0) & (f0 > 0)
    out[ok] = 12.0 * np.log2(f0[ok] / 100.0)
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True spans of `mask` as (start, end) frame indices."""
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _rms_db(x: np.ndarray) -> np.ndarray:
    if len(x) < WIN:
        return np.array([], dtype=float)
    r = librosa.feature.rms(y=x, frame_length=WIN, hop_length=HOP, center=False)[0]
    return 20.0 * np.log10(r + _EPS)


def _f0_st(x: np.ndarray, sr: int, return_db: bool = False):
    """Voiced F0 in semitones (NaN where unvoiced), 10 ms hop.

    `librosa.yin` (not pyin) for speed -- it is ~20x cheaper because it skips
    the HMM Viterbi decode, but it also returns a pitch for *every* frame,
    including silence. We supply the missing voicing decision ourselves:

      1. energy gate  : frame must be within 25 dB of the window's peak;
      2. range gate   : yin rails to fmin/fmax on aperiodic frames, so drop
                        anything pinned near either bound;
      3. median filter: 5-frame (50 ms) median kills isolated octave errors;
      4. outlier gate : discard frames > 12 st from the voiced median.
    """
    empty = np.array([], dtype=float)
    if len(x) < F0_FRAME:
        return (empty, empty) if return_db else empty
    try:
        f0 = librosa.yin(
            x, fmin=F0_MIN, fmax=F0_MAX, sr=sr,
            frame_length=F0_FRAME, hop_length=HOP, center=False,
        )
    except Exception:                      # pathological/degenerate segment
        return (empty, empty) if return_db else empty

    # RMS on identical framing -> frame-aligned with the yin output
    rms = librosa.feature.rms(
        y=x, frame_length=F0_FRAME, hop_length=HOP, center=False
    )[0]
    n = min(len(f0), len(rms))
    f0, rms = f0[:n], rms[:n]
    if n == 0:
        return (empty, empty) if return_db else empty

    db = 20.0 * np.log10(rms + _EPS)
    voiced = (
        (db >= db.max() - 25.0)
        & (f0 > F0_MIN * 1.05)
        & (f0 < F0_MAX * 0.95)
    )
    if not voiced.any():
        st = np.full(n, np.nan)
        return (st, db) if return_db else st

    k = 5 if n >= 5 else (3 if n >= 3 else 1)
    f0 = medfilt(f0, kernel_size=k)

    st = _hz_to_st(np.where(voiced, f0, np.nan))
    med = np.nanmedian(st)
    if np.isfinite(med):
        st[np.abs(st - med) > 12.0] = np.nan
    return (st, db) if return_db else st


def _fry_mask(st: np.ndarray, db: np.ndarray, tail_frames: int) -> np.ndarray:
    """Frames at the tail that look like creaky voice / vocal fry.

    Turn-final creak drops the glottal amplitude while yin keeps reporting a
    (garbage, often octave-halved) pitch. Those frames drag the terminal F0
    regression upward or downward at random, so we exclude them from the slope
    and keep only the clean modal-voice portion that precedes the fry.

    A tail frame is fry if it sits >= FRY_DROP_DB below the median level of the
    voiced frames in the window.
    """
    m = np.zeros(len(st), dtype=bool)
    if len(st) == 0 or len(db) != len(st):
        return m
    voiced = np.isfinite(st)
    if voiced.sum() < 2:
        return m
    ref_db = float(np.median(db[voiced]))
    tail = np.zeros(len(st), dtype=bool)
    tail[-tail_frames:] = True
    return tail & voiced & (db <= ref_db - FRY_DROP_DB)


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def extract_features(
    x: np.ndarray,
    sr: int,
    pause_start: float,
    pause_index: int = 0,
    prev_pause_start: float | None = None,
) -> np.ndarray:
    """Feature vector for one pause. Uses ONLY audio in [0, pause_start).

    Parameters
    ----------
    x, sr           : full waveform of the turn and its sample rate.
    pause_start     : seconds; the moment speech stops. Hard causal boundary.
    pause_index     : 0-based index of this pause within the turn (past info).
    prev_pause_start: pause_start of the previous pause, if any (past info).
    """
    # ---- CAUSAL CUT. Nothing below this line can see the future. ---------- #
    cut = x[: max(0, int(round(pause_start * sr)))]
    # ----------------------------------------------------------------------- #

    f = np.full(N_FEATURES, np.nan, dtype=np.float32)
    hop_s = HOP / sr

    # context features are always available, even for a degenerate cut
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    f[idx["elapsed_s"]] = pause_start
    f[idx["log_elapsed"]] = np.log1p(max(pause_start, 0.0))
    f[idx["pause_index"]] = pause_index
    f[idx["since_prev_pause_s"]] = (
        pause_start - prev_pause_start if prev_pause_start is not None else -1.0
    )

    if len(cut) < int(0.1 * sr):
        return f

    short = _tail(cut, sr, SHORT_S)
    long_ = _tail(cut, sr, LONG_S)
    ref = _tail(cut, sr, REF_S)

    # ---------------- pitch ------------------------------------------------ #
    st_ref, db_ref = _f0_st(ref, sr, return_db=True)
    if st_ref.size:
        med_ref = np.nanmedian(st_ref)          # reference BEFORE fry masking

        # exclude turn-final creak from everything downstream
        fry = _fry_mask(st_ref, db_ref, tail_frames=max(1, int(FRY_TAIL_S / hop_s)))
        n_short = int(SHORT_S / hop_s)
        n_long = int(LONG_S / hop_s)
        voiced_tail = np.isfinite(st_ref[-n_short:])
        f[idx["f0_fry_frac"]] = (
            float(fry[-n_short:].sum() / voiced_tail.sum()) if voiced_tail.sum() else 0.0
        )
        st_clean = st_ref.copy()
        st_clean[fry] = np.nan

        st_s = st_clean[-n_short:]
        st_l = st_clean[-n_long:]
        voiced_s = st_s[np.isfinite(st_s)]

        f[idx["f0_slope_st_per_s"]] = _slope(st_s, hop_s)
        f[idx["f0_slope_long"]] = _slope(st_l, hop_s)
        f[idx["f0_voiced_frac"]] = float(np.isfinite(st_s).mean())
        if voiced_s.size:
            f[idx["f0_final_st"]] = float(voiced_s[-1])
            f[idx["f0_last_minus_med_st"]] = float(voiced_s[-1] - med_ref)
            f[idx["f0_range_st"]] = float(voiced_s.max() - voiced_s.min())
            f[idx["f0_std_st"]] = float(voiced_s.std())
            f[idx["f0_delta_last_two"]] = (
                float(voiced_s[-1] - voiced_s[-2]) if voiced_s.size >= 2 else 0.0
            )

    # ---------------- energy ----------------------------------------------- #
    db_ctx = _rms_db(cut)
    db_l = _rms_db(long_)
    if db_l.size >= 3:
        n_short = min(len(db_l), int(SHORT_S / hop_s))
        db_s = db_l[-n_short:]
        p90 = float(np.percentile(db_ctx, 90)) if db_ctx.size else float(db_l.max())
        floor_db = p90 - 20.0

        f[idx["rms_slope_db_per_s"]] = _slope(db_s, hop_s)
        f[idx["rms_slope_long"]] = _slope(db_l, hop_s)
        f[idx["rms_last_db"]] = float(db_l[-1])
        f[idx["rms_last_minus_p90"]] = float(db_l[-1] - p90)
        n30 = min(len(db_l), int(0.30 / hop_s))
        f[idx["rms_drop_300ms"]] = float(db_l[-n30] - db_l[-1])
        f[idx["rms_std_db"]] = float(db_s.std())
        f[idx["rms_frac_below_20db"]] = float((db_s < floor_db).mean())

        # ---------------- timing / rhythm ---------------------------------- #
        loud = db_l >= floor_db
        runs = [(a, b) for a, b in _runs(loud) if (b - a) * hop_s >= 0.04]
        f[idx["voiced_frac_long"]] = float(loud.mean())
        f[idx["run_rate_hz"]] = len(runs) / max(len(db_l) * hop_s, _EPS)
        if runs:
            durs = np.array([(b - a) * hop_s for a, b in runs])
            f[idx["final_run_s"]] = float(durs[-1])
            f[idx["final_run_ratio"]] = float(durs[-1] / (durs.mean() + _EPS))
        else:
            f[idx["final_run_s"]] = 0.0
            f[idx["final_run_ratio"]] = 0.0

    # ---------------- spectral (trailing 500 ms) --------------------------- #
    if len(short) >= WIN:
        kw = dict(n_fft=512, hop_length=HOP, center=False)
        S = np.abs(librosa.stft(short, **kw)) + _EPS
        f[idx["zcr_mean"]] = float(
            librosa.feature.zero_crossing_rate(
                short, frame_length=WIN, hop_length=HOP, center=False
            ).mean()
        )
        f[idx["spec_centroid_last"]] = float(
            librosa.feature.spectral_centroid(S=S, sr=sr).mean()
        )
        f[idx["spec_rolloff_last"]] = float(
            librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.90).mean()
        )
        f[idx["spec_flatness_last"]] = float(
            librosa.feature.spectral_flatness(S=S).mean()
        )

        mel = librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=40, fmax=sr // 2)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC)
        for i in range(N_MFCC):
            f[idx[f"mfcc{i}_mean"]] = float(mfcc[i].mean())
            f[idx[f"mfcc{i}_std"]] = float(mfcc[i].std())
        if mfcc.shape[1] >= 9:
            d = librosa.feature.delta(mfcc, width=min(9, mfcc.shape[1] // 2 * 2 + 1))
            for i in range(6):
                f[idx[f"dmfcc{i}_mean"]] = float(d[i].mean())
        else:
            for i in range(6):
                f[idx[f"dmfcc{i}_mean"]] = 0.0

    # ---------------- multi-scale slopes ----------------------------------- #
    if st_ref.size:
        for tag, secs in (("250ms", 0.25), ("1s", 1.0), ("2s", 2.0)):
            k = int(secs / hop_s)
            f[idx[f"f0_slope_{tag}"]] = _slope(st_clean[-k:], hop_s)
    db_r3 = _rms_db(ref)
    if db_r3.size >= 3:
        for tag, secs in (("250ms", 0.25), ("1s", 1.0), ("2s", 2.0)):
            k = int(secs / hop_s)
            f[idx[f"rms_slope_{tag}"]] = _slope(db_r3[-k:], hop_s)
        k25 = int(0.25 / hop_s)
        a, b = db_r3[-k25:], db_r3[-2 * k25:-k25]
        f[idx["energy_ratio_250_500"]] = float(a.mean() - b.mean()) if b.size else 0.0

    # ---------------- voice quality (last 300 ms) -------------------------- #
    vq = _tail(cut, sr, 0.30)
    if len(vq) >= WIN:
        P = np.abs(librosa.stft(vq, n_fft=512, hop_length=HOP, center=False)) ** 2 + _EPS
        freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
        lo = P[(freqs >= 80) & (freqs < 1000)].mean()
        hi = P[(freqs >= 1000) & (freqs < 4000)].mean()
        f[idx["spec_tilt_db"]] = float(10 * np.log10(lo / hi))
        # harmonic-to-noise proxy: peak of the normalised autocorrelation
        v = vq - vq.mean()
        ac = np.correlate(v, v, mode="full")[len(v) - 1:]
        if ac[0] > 0:
            ac = ac / ac[0]
            k_lo, k_hi = int(sr / F0_MAX), min(int(sr / F0_MIN), len(ac) - 1)
            if k_hi > k_lo:
                r = float(np.clip(ac[k_lo:k_hi].max(), 1e-4, 0.9999))
                f[idx["hnr_proxy_db"]] = float(10 * np.log10(r / (1 - r)))

    if st_ref.size:
        tail_v = st_clean[-int(SHORT_S / hop_s):]
        tail_v = tail_v[np.isfinite(tail_v)]
        if tail_v.size >= 3:
            hz = 100.0 * 2 ** (tail_v / 12.0)
            f[idx["jitter_pct"]] = float(100 * np.mean(np.abs(np.diff(hz))) / hz.mean())
    if db_l.size >= 3:
        f[idx["shimmer_db"]] = float(np.mean(np.abs(np.diff(db_l[-int(SHORT_S / hop_s):]))))
        loud_l = db_l >= (float(np.percentile(db_ctx, 90)) - 20.0 if db_ctx.size
                          else db_l.max() - 20.0)
        rr = [(b - a) * hop_s for a, b in _runs(loud_l) if (b - a) * hop_s >= 0.04]
        if rr:
            f[idx["voiced_run_final_s"]] = float(rr[-1])
            f[idx["run_dur_mean"]] = float(np.mean(rr))
            f[idx["run_dur_std"]] = float(np.std(rr))
            f[idx["run_dur_max"]] = float(np.max(rr))

    # ---------------- turn-level signature (all audio so far) -------------- #
    if len(cut) >= WIN:
        Sc = np.abs(librosa.stft(cut, n_fft=512, hop_length=HOP, center=False)) + _EPS
        melc = librosa.feature.melspectrogram(S=Sc ** 2, sr=sr, n_mels=40, fmax=sr // 2)
        mfc = librosa.feature.mfcc(S=librosa.power_to_db(melc), n_mfcc=8)
        for i in range(8):
            f[idx[f"turn_mfcc{i}"]] = float(mfc[i].mean())
    if db_ctx.size:
        p90c = float(np.percentile(db_ctx, 90))
        f[idx["turn_rms_p90"]] = p90c
        loud_c = db_ctx >= p90c - 20.0
        f[idx["turn_voiced_frac"]] = float(loud_c.mean())
        rc = [(b - a) for a, b in _runs(loud_c) if (b - a) * hop_s >= 0.04]
        f[idx["turn_run_rate_hz"]] = len(rc) / max(len(db_ctx) * hop_s, _EPS)
    # pitch signature: subsample long turns so cost stays flat
    sig = cut if len(cut) <= 6 * sr else np.concatenate(
        [cut[: 3 * sr], cut[-3 * sr:]])
    st_sig = _f0_st(sig, sr)
    if st_sig.size:
        vv = st_sig[np.isfinite(st_sig)]
        if vv.size:
            f[idx["turn_f0_med_st"]] = float(np.median(vv))
            f[idx["turn_f0_iqr_st"]] = float(np.percentile(vv, 75) - np.percentile(vv, 25))

    return f


def extract_from_file(path, pause_start, pause_index=0, prev_pause_start=None):
    """Convenience wrapper: load a wav and extract one pause's features."""
    x, sr = load_audio(path)
    return extract_features(x, sr, pause_start, pause_index, prev_pause_start)
