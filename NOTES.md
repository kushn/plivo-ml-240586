# NOTES

The model scores each pause from 60 causal features covering terminal pitch
(YIN F0 in semitones, slope over the last 500 ms and 1.5 s, final value against a
3 s local speaker reference), energy decay into the pause, final-syllable
lengthening and syllable rate, position in the turn, and trailing MFCCs — all
computed strictly from `audio[0 : pause_start]`.

The strongest single cue is terminal pitch movement, but it points the opposite
way in the two languages: Hindi turn ends fall (median −2.60 st/s vs +0.22 for
holds) while English turn ends in this corpus rise (+1.97 vs −3.76), so the MFCC
block earns its place by acting as an implicit language identifier that lets the
trees condition the sign of that rule — pruning it dropped held-out AUC from
0.705 to 0.650.

Scored per folder the silence-only baseline is 1600 ms on English but already
850 ms on Hindi, so the honest result is English 1600 -> 1130 ms (-29%) and Hindi
at parity with the timer; what finally moved English was weighting training rows
by what a mistake costs, since the scorer charges a false cutoff only on holds
longer than the action delay and short hesitations - 80% of the negatives - are
free to fire on.

It fails on Hindi, where beating 850 ms means separating true ends from the ~12
turns whose holds exceed a 700 ms delay and the features only reach 0.68 AUC on
that specific contrast; it also fails on long
mid-sentence hesitations where the speaker trails off with turn-final prosody and
then resumes, and on turns whose first pause already looks complete, since
`pause_index` biases early pauses toward `hold`. It also cannot use lexical
completion at all, which is the signal a human listener actually relies on.

Multi-scale slope windows, voice-quality features (spectral tilt, jitter,
shimmer, HNR) and an explicit turn-level language signature were all tried and
all failed to help (+0.004 AUC), so with one more day I would change the
objective rather than the features: train a pairwise ranker on exactly the
(true end, hold longer than the action delay) pairs the metric actually scores,
and — first, before any of that — listen to those dozen Hindi long holds, which
is the one input none of the seven completed runs has had.
