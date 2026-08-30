# SIM: Search-based Interest Model

Tags: sequence, attention, architecture, user-history, long-sequence

Source: Pi et al., "Search-based User Interest Modeling with Lifelong Sequential
Behavior Data", CIKM 2020. arXiv:2006.05639

## What it is

DIN applied to very long histories, made tractable by a retrieval step. Attention over a
user's full lifetime of behaviour is too expensive, so SIM first *retrieves* a small
relevant subset of the history, then attends over only that subset.

Two retrieval modes:

- **Hard search**: keep only history items sharing a categorical attribute with the
  candidate — same category, same author, same tag. Requires no training, just an index.
- **Soft search**: retrieve by embedding similarity to the candidate, using an index
  trained jointly with an auxiliary CTR task.

Hard search is what most deployments use, because it is essentially free and the
category signal is strong.

## Why it helps a ranking metric

Same mechanism as DIN — a candidate-dependent user vector varies within a user's list and
therefore can reorder it — but reaching further back. The additional argument is
signal-to-noise: attending over 1000 history items dilutes the relevant ones, whereas
retrieving the 30 items in the same category as the candidate concentrates them.

For a short-video dataset where users have hundreds to thousands of logged interactions,
plain DIN's fixed recent window of 20-50 items discards most of the history, and the
recent window is dominated by whatever the user happened to be shown lately rather than
their durable interests.

## How to implement

Hard search is the version to implement first, and in numpy it is an index build plus a
gather:

1. Build, per (user, category), the list of item ids that user interacted with.
2. At scoring time, look up the candidate's category, fetch that user's list for that
   category, truncate to the most recent L.
3. Attend over the retrieved set exactly as in DIN.

The same strict-time-ordering rule as DIN applies: the retrieved set must contain only
interactions before the row's timestamp. With a per-(user, category) list sorted by time,
this is a binary search for the timestamp followed by a backward slice.

Precompute the retrieval for every row once, before training, rather than inside the
training loop — it does not depend on model parameters, so it is a fixed input like any
other feature.

Author id is often a better retrieval key than category in short-video data: category
partitions are coarse, while "has this user watched this creator before" is a sharp,
high-precision signal that also varies within a user's impression list.

## When it will not help

- If per-user history is short, retrieval returns almost nothing and SIM degenerates to
  DIN with extra machinery. Measure the history-length distribution first.
- If the retrieval key is too coarse (a category with only a handful of distinct values),
  the retrieved set is barely more relevant than a random slice of history.
- The soft-search variant needs a jointly trained index and is substantially more work.
  Do not start there.
