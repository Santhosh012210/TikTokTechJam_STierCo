# BPR: Bayesian Personalized Ranking

Tags: loss, pairwise, ranking-objective

Source: Rendle, Freudenthaler, Gantner, Schmidt-Thieme, "BPR: Bayesian Personalized
Ranking from Implicit Feedback", UAI 2009. arXiv:1205.2618

## What it is

A pairwise ranking objective for implicit feedback. Instead of asking "what is the
probability this item is positive?", it asks "is this positive item scored above that
negative item?" For a user `u` with a positive item `i` and a negative item `j`, the
loss on that triple is

    L = -log(sigmoid(s_ui - s_uj))

Only the *difference* of the two scores enters the loss. The model is never asked to
produce a calibrated probability, only a correct ordering.

## Why it helps a ranking metric

Pointwise logloss optimises calibration: it wants every score to approach the true
click probability. GAUC and nDCG do not care about calibration at all — they read only
the order of scores inside one user's impression list. That is a mismatch: pointwise
loss spends model capacity getting absolute probabilities right, which the metric then
discards.

BPR's loss is exactly a smooth surrogate for pairwise ordering accuracy, and GAUC *is*
pairwise ordering accuracy averaged within users. Optimising BPR optimises something
monotonically related to the metric being scored.

A second effect matters on within-user ranking specifically: any term that is constant
within a user cancels in `s_ui - s_uj`. A user bias, or any pure user-side first-order
feature, contributes literally nothing to the gradient. Pointwise loss will happily
spend capacity fitting those terms; BPR cannot, so the capacity goes where it counts.

## How to implement

The critical detail: **negatives must be sampled from the same user's impressions.**
Sampling a negative from a different user optimises cross-user ordering, which the
metric never measures, and the resulting model can score worse than pointwise.

Build the pair index once, before training:

    # group row indices by user, split into positive and negative pools
    pos_by_user = {}   # user -> array of row indices with label 1
    neg_by_user = {}   # user -> array of row indices with label 0
    # keep only users that have at least one of each; the rest cannot form a pair
    # (this mirrors GAUC, which also skips all-positive and all-negative users)

Per training step, sample a batch of positives, then one negative per positive from the
same user's negative pool:

    pos_idx = rng.choice(all_pairable_positives, size=B)
    users   = user_of_row[pos_idx]
    neg_idx = np.array([rng.choice(neg_by_user[u]) for u in users])

That per-row `rng.choice` is slow in a Python loop. A vectorised version: store the
negative pools in a ragged flat array with per-user offsets and lengths, then

    offsets = neg_offset[users]
    counts  = neg_count[users]
    neg_idx = neg_flat[offsets + (rng.random(B) * counts).astype(np.int64)]

The gradient is where most implementations go wrong. With `d = s_pos - s_neg` and
`g = -sigmoid(-d)` (the derivative of `-log(sigmoid(d))` with respect to `d`), the
gradient with respect to any parameter `w` is

    dL/dw = g * (d s_pos / dw - d s_neg / dw)

So for an FM you compute the *same* per-example gradient you already compute for
pointwise, then accumulate it with `+g` at the positive row's features and `-g` at the
negative row's features. Concretely, if `grad_fm(X)` returns the per-row parameter
gradient, the BPR update scatters `g[:, None] * grad_fm(X_pos)` and
`-g[:, None] * grad_fm(X_neg)` into the same accumulator with `np.add.at`.

Note that a feature shared by both rows of a pair (the user embedding, for example)
receives `+g` and `-g` into the same slot and cancels — which is the within-user
cancellation described above, appearing naturally in the arithmetic.

## When it will not help

- If negatives are sampled globally rather than within-user, expect no gain or a loss.
- Users with zero positives or zero negatives produce no pairs. They are unrankable by
  any model and are exactly the users GAUC already excludes, so dropping them from
  training is correct, not a data loss.
- BPR converges to a different scale of scores than pointwise. Do not compare raw
  score distributions across the two; compare only the metric.
