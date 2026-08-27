# Target encoding and historical-rate features

Tags: features, feature-engineering, leakage, gbdt

Source: Micci-Barreca, "A preprocessing scheme for high-cardinality categorical
attributes", SIGKDD Explorations 2001. Standard practice in Kaggle-style tabular
pipelines.

## What it is

Replace a high-cardinality categorical value with a statistic of the target computed over
rows sharing that value — most often the smoothed historical positive rate:

    encoded(c) = (sum_of_positives(c) + m * prior) / (count(c) + m)

`prior` is the global positive rate and `m` is a smoothing constant, typically 10-100.
Smoothing pulls rare categories towards the global mean so a category seen twice with two
positives does not encode as 1.0.

## Why it helps a ranking metric

It converts an ID that a tree cannot split on, and that an embedding model can only learn
slowly from sparse observations, into a single dense informative number.

For a within-user ranking metric, note which encodings can help: **item-side encodings
vary within a user's list and can reorder it; user-side encodings cannot.** An item's
historical positive rate is a usable feature. A user's historical positive rate is
constant across that user's rows and contributes nothing on its own — it is only useful
crossed with something item-side.

This is the main mechanism by which a GBDT becomes competitive on ID-heavy recommendation
data, since trees cannot consume raw IDs.

## How to implement

The implementation is trivial. The correctness is not, and target encoding is one of the
most reliable ways to produce a validation score that does not survive contact with a
test set.

**Rule 1: compute encodings on training data only.** Statistics computed over a span that
includes validation rows leak validation labels into training features.

**Rule 2: respect time ordering.** For a time-split dataset, an encoding applied to a row
must use only data from before that row. Computing an item's rate over the whole training
period and applying it to rows inside that same period lets a row contribute to its own
feature.

**Rule 3: out-of-fold encoding for training rows.** The standard fix for rule 2 within a
single period — split training into K folds, encode each fold using statistics from the
other K-1. For time-split data, expanding-window encoding (each row uses only strictly
earlier rows) is more faithful to how the model will be used.

    # expanding window, per item, rows sorted by time
    cum_pos   = np.cumsum(y) - y          # positives strictly before this row
    cum_count = np.arange(len(y)) - 0     # rows strictly before this row
    encoded   = (cum_pos + m * prior) / (cum_count + m)

Useful encodings for a short-video ranking log, all item-side or crossed:

- item positive rate, item impression count
- author positive rate, author impression count
- (user, author) interaction count — crossed, so it varies within a user's list
- item mean watch-time percentile

## When it will not help

- A model that already learns ID embeddings gets less from target encoding, because the
  embedding is a richer version of the same information. The gain is largest for trees.
- Rare categories encode almost entirely to the prior and carry no information no matter
  how the smoothing is tuned.
- Any leak makes the feature look excellent on validation and worthless on test. If a
  target-encoded feature produces a suspiciously large jump, audit the time ordering
  before believing it.
