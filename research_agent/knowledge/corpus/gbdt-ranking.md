# GBDT learning-to-rank (LambdaMART in XGBoost / LightGBM)

Tags: gbdt, ranking-objective, industry, architecture

Source: Burges, MSR-TR-2010-82 (LambdaMART). Implementations: XGBoost `rank:pairwise` /
`rank:ndcg`, LightGBM `lambdarank`.

## What it is

Gradient-boosted decision trees trained with a ranking objective. The libraries take a
`group` array giving the size of each query's candidate list, and optimise a
LambdaRank-style objective within each group.

For a within-user ranking task, **the group is the user**: each user's impression list is
one "query".

## Why it helps a ranking metric

Two arguments, both practical rather than theoretical.

First, GBDTs are the strongest model class on tabular features with limited data, and
frequently beat neural approaches on datasets of around a million rows. They need no
embedding table, no learning-rate schedule, and no epoch tuning.

Second, `rank:ndcg` / `lambdarank` gives metric-aligned optimisation *for free* — the
objective alignment that BPR or LambdaRank would take real implementation effort to build
by hand is a single parameter here.

The catch: trees cannot use raw high-cardinality ID features. `user_id` and `item_id`
with thousands of distinct values are useless as split candidates. This means GBDT is not
a drop-in replacement for an ID-embedding model; it is a different modelling approach
requiring engineered numeric features — historical rates, counts, aggregates.

## How to implement

Sort rows by user, then build the group array:

    # rows must be sorted by user for group semantics to be correct
    order = np.argsort(user_ids, kind="stable")
    groups = np.bincount(user_ids[order])      # impressions per user, in user order

    dtrain = xgb.DMatrix(X[order], label=y[order])
    dtrain.set_group(groups[groups > 0])
    params = {"objective": "rank:ndcg", "eval_metric": "ndcg@5", "eta": 0.1, "max_depth": 6}

Feature engineering is where the work is. Candidate features that vary within a user (a
requirement — see the note on within-user ranking):

- item historical positive rate, smoothed towards the global mean
- item impression count, log-scaled
- author historical positive rate
- video duration and duration bucket
- user-item affinity: has this user interacted with this author before, count and rate

**Every one of these must be computed on training data only, with strict time ordering.**
Computing an item's positive rate over the full data including validation rows leaks the
label. This is the single most common way GBDT recommender pipelines produce an inflated
validation score that collapses on test.

A hybrid is often the strongest option: take the FM's predicted score as one input
feature to the GBDT, alongside the engineered features. That combines the ID-embedding
signal the trees cannot learn with the tabular signal the FM cannot express.

## When it will not help

- If the libraries are not installed and cannot be added, this is unavailable — check
  before planning around it.
- With only raw ID features and no engineered aggregates, trees have almost nothing to
  split on and will underperform the embedding model badly.
- Target-encoded features computed without strict time ordering leak. If a GBDT
  validation score jumps implausibly, suspect leakage before celebrating.
