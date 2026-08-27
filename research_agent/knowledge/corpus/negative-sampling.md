# Negative sampling strategies

Tags: training, sampling, loss, ranking-objective

Source: Rendle et al., UAI 2009 (uniform, in BPR). Hard-negative and popularity-based
variants: Rendle and Freudenthaler, WSDM 2014; Chen et al., "Sampling-bias-corrected
neural modeling", RecSys 2019.

## What it is

Any pairwise or listwise ranking loss needs negatives to contrast positives against. How
those negatives are chosen changes what the model learns, often more than the choice of
loss function.

Main strategies:

- **Uniform within group**: sample uniformly from the same user's observed negative
  impressions.
- **Popularity-weighted**: sample proportional to item frequency raised to a power,
  making popular items more likely negatives, which counteracts popularity bias.
- **Hard negatives**: sample from items the model currently scores highly but which are
  labelled negative. Strongest gradient signal per sample.
- **In-batch negatives**: reuse other rows in the batch as negatives. Nearly free, but
  they come from other users.

## Why it helps a ranking metric

For a within-user metric the choice is largely settled by the structure of the task:
negatives must come from the same user's impression list. The metric only ever compares
items inside one user's list, so a cross-user comparison optimises pairs the metric never
scores.

This makes in-batch negatives — the default in most retrieval code, and the cheapest
option — actively wrong here unless batches are constructed per user. It is a common way
a correctly implemented ranking loss produces no gain.

Within that constraint, hard negatives matter because most within-user pairs are easy.
Once a model reliably separates obvious positives from obvious negatives, uniform sampling
draws mostly pairs it already gets right, and their gradients are near zero.

## How to implement

Uniform within-user, vectorised with a ragged flat array (a Python-level `rng.choice` per
row is the bottleneck otherwise):

    # built once: negatives grouped by user, flattened
    neg_flat, neg_offset, neg_count = build_ragged(neg_rows_by_user)

    offsets = neg_offset[users]
    counts  = neg_count[users]
    picks   = (rng.random(len(users)) * counts).astype(np.int64)
    neg_idx = neg_flat[offsets + picks]

Hard negatives, cheaply approximated without rescoring the corpus every step: sample `M`
candidates uniformly (M = 4 or 8), score them, keep the highest-scoring one.

    cand = neg_flat[offsets[:, None] + (rng.random((B, M)) * counts[:, None]).astype(np.int64)]
    scores = model.predict(X[cand.ravel()]).reshape(B, M)
    neg_idx = cand[np.arange(B), scores.argmax(1)]

That costs `M` extra forward passes and no extra backward passes.

Ramp hard negatives in gradually. Training on hard negatives from step 0 is unstable
because early "hard" negatives are just noise from an untrained model. A common schedule
starts uniform and raises `M` over the first few epochs.

## When it will not help

- If negatives are drawn across users, expect no gain regardless of the loss. Check this
  first when a ranking loss underperforms.
- Hard negatives on a noisy label can amplify label noise — a mislabelled positive looks
  exactly like a hard negative.
- Users with no observed negatives cannot form pairs. They are also excluded from GAUC,
  so dropping them from training costs nothing measurable.
