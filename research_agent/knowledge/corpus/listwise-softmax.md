# Listwise softmax over a user's impressions

Tags: loss, listwise, ranking-objective

Source: Cao, Qin, Liu, Tsai, Li, "Learning to Rank: From Pairwise Approach to Listwise
Approach", ICML 2007 (ListNet). The single-positive special case is the sampled-softmax
loss used throughout industrial retrieval.

## What it is

Treat one user's impression list as a single training example. Score every item in the
list, take a softmax over those scores, and maximise the log-probability mass on the
positive items:

    p = softmax(s_1 ... s_n)          # over one user's n impressions
    L = -sum over positives of log(p_i)

Where pairwise loss looks at two items at a time, listwise looks at the whole list at
once, so a positive is pushed above *all* negatives in the list simultaneously rather
than one sampled negative at a time.

## Why it helps a ranking metric

nDCG@5 is a position-sensitive metric: it cares most about the very top of each user's
list. Pairwise loss treats all inversions equally — swapping ranks 1 and 2 costs the
same as swapping 40 and 41. Softmax loss weights gradients by `p_i`, which concentrates
on the items the model currently ranks highly, so its gradient pressure is naturally
biased towards the top of the list where nDCG@5 is decided.

Like BPR, it is shift-invariant within a user: adding a constant to every score in a
list leaves the softmax unchanged, so user-constant terms contribute no gradient.

## How to implement

Group rows by user once, at load time. Because impression list lengths vary, the clean
numpy approach is a flat array plus segment boundaries rather than a padded matrix.

Segment-wise softmax with the standard max-subtraction for numerical stability, using
`np.maximum.at` and `np.add.at` to reduce within each user's segment:

    seg = user_index_of_row          # int array, rows sorted by user
    m = np.full(n_users, -np.inf)
    np.maximum.at(m, seg, scores)    # per-user max
    e = np.exp(scores - m[seg])
    Z = np.zeros(n_users)
    np.add.at(Z, seg, e)             # per-user partition function
    p = e / Z[seg]

The gradient of `L` with respect to `scores` is beautifully simple:

    dL/ds = p - y_normalised

where `y_normalised` is the label vector divided by that user's positive count (so each
user contributes equally regardless of how many positives they have). With one positive
per user this reduces to the familiar `p - onehot`.

Then backpropagate `dL/ds` into the model exactly as you would any per-row score
gradient — for FM, scatter it through the same `np.add.at` path the pointwise loss uses.

Batching: batch by *user*, not by row. A batch is some number of complete impression
lists. Truncating a user's list across a batch boundary silently changes the loss,
because the partition function would then be computed over a partial list.

## When it will not help

- If lists are truncated or split across batches, the normalisation is wrong and results
  will be erratic. Batch by user.
- Users with very long impression lists dominate the loss unless you normalise per user.
  The `y_normalised` division above handles the positive count; consider also whether
  users with 500 impressions should outweigh users with 5.
- Sensitive to a temperature/scale parameter in a way BPR is not. If scores are large,
  the softmax saturates and gradients vanish. Consider dividing scores by a learned or
  fixed temperature.
