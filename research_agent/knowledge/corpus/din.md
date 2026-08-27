# DIN: Deep Interest Network

Tags: sequence, attention, architecture, user-history

Source: Zhou et al., "Deep Interest Network for Click-Through Rate Prediction",
KDD 2018. arXiv:1706.06978

## What it is

A CTR model that represents a user by *attending* over their behaviour history with
respect to the candidate item, rather than by a single static user embedding.

The standard approach pools a user's history into one fixed vector — usually a mean of
the embeddings of items they interacted with. DIN's observation is that this is wasteful:
a user who watched cooking videos and basketball videos gets one blended vector, and that
blend is used whether the candidate is a recipe or a game highlight. DIN instead computes
attention weights between the candidate item and each history item, then pools with those
weights:

    weights = a(e_history_j, e_candidate)      # small MLP, one scalar per history item
    u = sum_j weights_j * e_history_j          # candidate-specific user vector

The user representation therefore changes for every candidate being scored.

## Why it helps a ranking metric

This is the property that matters for within-user ranking. A static user embedding is
*constant within a user*, so under a ranking metric that only reads intra-user order, it
contributes nothing — it shifts every score in the list by the same amount. Any model
whose user-side contribution is a fixed vector is spending parameters on something the
metric cannot see.

DIN's user vector is a function of the candidate, so it varies within the user's list.
That is the whole point: it converts user-side information from a rank-invariant constant
into a term that actually reorders the list.

If the current feature set contains no behaviour sequence at all, this is untouched
territory rather than an incremental tweak — the model currently has no way to express
"this user has watched things like this before".

## How to implement

Building the history is the bulk of the work and it is where correctness bugs live.

**Leakage rule: a row's history must contain only interactions strictly before that
row's timestamp.** Building history from the whole training split and applying it to
training rows leaks the future into the past and produces a validation score that will
not reproduce. Sort by (user, timestamp) and take a backward window.

    # for each row, the previous L items that user interacted with, before this row
    # L = 20..50 is typical; longer needs SIM-style retrieval instead
    history[i] = last L item_ids of user u strictly before time t_i

Store as a padded `(n_rows, L)` int array plus a `(n_rows,)` length array; pad with a
reserved index 0 and mask it out of the pooling.

The attention unit in the paper is a small MLP over
`[e_hist, e_cand, e_hist - e_cand, e_hist * e_cand]` — the difference and product terms
carry the interaction signal and matter more than the depth of the MLP. A pure-numpy
implementation can start much simpler and still capture most of the gain:

    # dot-product attention, no learned parameters at all
    scores = (E_hist * e_cand[:, None, :]).sum(-1) / sqrt(k)
    scores = where(mask, scores, -inf)
    w = softmax(scores, axis=1)
    u = (w[:, :, None] * E_hist).sum(1)

Then feed `u` into the model alongside the existing features — for an FM, the cleanest
entry point is to add `dot(u, e_cand)` to the logit, which is one extra scalar term.

**Do not normalise the attention weights to sum to 1 in the paper's formulation.** DIN
deliberately omits the softmax so that the magnitude of the user vector reflects the
*intensity* of matching history, not just its direction. Both variants are worth trying;
the softmax version is easier to make numerically stable.

## When it will not help

- If the history window is built with any lookahead, results will be inflated and will
  not transfer to validation. Check this first if the gain looks too large.
- Users with almost no history get a near-empty attention pool and fall back to whatever
  the rest of the model does. Check the distribution of per-user interaction counts
  before assuming the average user has a usable sequence.
- Adding history as a *mean-pooled* vector without attention reproduces the static-user
  problem and will contribute little under a within-user metric. The attention is not an
  optional refinement here; it is the part that makes the term rank-relevant.
