# SASRec: Self-Attentive Sequential Recommendation

Tags: sequence, attention, architecture, transformer, user-history

Source: Kang and McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018.
arXiv:1808.09781

## What it is

A transformer decoder over a user's interaction sequence. Each position attends over all
previous positions with causal masking, and the representation at position t is used to
predict the item at position t+1. Where DIN attends from the *candidate* to the history,
SASRec builds a representation of the sequence itself.

## Why it helps a ranking metric

It captures order and transition structure that pooling methods discard: "this user just
watched three cooking videos in a row" is a different state from "this user watched three
cooking videos scattered over a month", and only a sequence model can tell them apart.

For within-user ranking, the sequence representation is user-constant on its own — it does
not depend on the candidate — so it must be *crossed* with the candidate to affect the
ordering, typically as a dot product between the sequence state and the candidate
embedding. That dot product is the term that varies within the list.

## How to implement

Honest assessment: this is the heaviest option in this corpus. A correct implementation
needs multi-head attention, positional embeddings, causal masking, layer norm, and
residual connections. In pure numpy with hand-written backprop, that is a large amount of
error-prone code, and the backward pass through attention is where it will go wrong.

If a deep learning framework is available, it is a few dozen lines and the framework
handles the gradient. If the environment is numpy-only, prefer DIN: it captures much of
the benefit with a fraction of the implementation risk, because its attention has no
learned parameters in the simplified form and its backward pass is a single scatter.

A middle option that is tractable in numpy: single-head, single-layer self-attention with
no feed-forward block and no layer norm. That keeps the "representation depends on the
whole sequence" property while making the gradient writable by hand.

Sequence construction follows the same strict-time-ordering rule as DIN — position t may
see only positions before t. The causal mask enforces this within the attention; the
sequence itself must additionally not extend past the row's timestamp.

## When it will not help

- Against a strong FM on a 1M-row dataset, transformer capacity is usually not the
  binding constraint. If capacity ablations already showed embedding dimension does not
  matter, that is evidence the model is not underfitting.
- Cold users with sequences of length 0-2 get essentially no signal.
- Implementation cost is high relative to the alternatives in this corpus. Reach for it
  after the cheaper objective-alignment and attention methods have been measured.
