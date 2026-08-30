# LambdaRank and LambdaMART

Tags: loss, listwise, ranking-objective, ndcg, gbdt

Source: Burges, "From RankNet to LambdaRank to LambdaMART: An Overview",
Microsoft Research Technical Report MSR-TR-2010-82, 2010.

## What it is

A modification to pairwise ranking that makes the gradient aware of the *metric*. Start
from RankNet's pairwise loss, then multiply each pair's gradient by the absolute change
in the target metric that would result from swapping that pair:

    lambda_ij = -sigmoid(-(s_i - s_j)) * |delta_nDCG(i, j)|

`delta_nDCG(i, j)` is the change in nDCG from exchanging the positions of items i and j
while holding everything else fixed. The insight is that you cannot differentiate nDCG
(it is a step function of the ranks), but you can write down what the gradient *should*
be, and it turns out that is enough — the lambdas are the gradient of a loss whose
existence was only proved later.

LambdaMART is the same idea with gradient-boosted trees as the model instead of a neural
net; it is the standard strong baseline for learning-to-rank.

## Why it helps a ranking metric

This is the most directly metric-aligned option available. Plain pairwise loss treats
every inversion as equally bad. nDCG@5 does not: an inversion between positions 1 and 2
matters enormously and an inversion between positions 30 and 31 does not matter at all
(both are outside the cutoff). LambdaRank encodes precisely that weighting into the
gradient, so the model spends its capacity where the metric is actually decided.

For a metric that is `mean(GAUC, nDCG@5)`, note that the two halves want different
things: GAUC weights all inversions equally (it is plain pairwise accuracy) while nDCG@5
weights the top. A lambda weighting derived from nDCG@5 alone will trade a little GAUC
for more nDCG. That trade may or may not be net positive — it is worth measuring both
metrics separately, not just the mean.

## How to implement

Per user, per training step:

1. Score the user's impressions and sort descending to get current ranks.
2. Compute the ideal DCG for that user once (it depends only on labels, so cache it).
3. For every (positive, negative) pair, compute the delta:

       # gains for binary labels: 2^rel - 1 = rel
       # discounts: 1 / log2(rank + 2) for 0-based rank
       delta_dcg = (gain_i - gain_j) * (discount_at_rank_i - discount_at_rank_j)
       delta_ndcg = abs(delta_dcg) / ideal_dcg

4. Accumulate per-item lambdas: item i receives `+lambda_ij`, item j receives
   `-lambda_ij`, summed over all pairs it participates in.
5. Backpropagate the per-item lambda as the score gradient.

With binary labels the gain simplifies: `2^1 - 1 = 1` for a positive and `0` for a
negative, so `gain_i - gain_j` is just 1 for every positive/negative pair. That reduces
delta_ndcg to `|discount_i - discount_j| / ideal_dcg`, which is cheap.

Vectorising step 3 for one user with `P` positives and `N` negatives is a `P x N`
outer-product-shaped computation — fine for the impression-list sizes in a logged
ranking dataset, but do not build the full pair matrix across all users at once.

Important subtlety: ranks change every time the model updates, so lambdas must be
recomputed from the *current* scores each epoch. This is not a fixed target you can
precompute once.

If `nDCG@5` specifically is the target, zeroing the lambda for pairs where both items
sit below rank 5 in the current ordering focuses the computation further, at the cost of
some gradient signal early in training when the ordering is still random.

## When it will not help

- It is strictly more expensive per step than BPR. If BPR has not been tried yet, try
  that first — it captures most of the objective-alignment gain for much less work.
- The lambda weighting assumes the current ranking is roughly meaningful. Very early in
  training, when scores are near-random, the deltas are noise. Warm-starting from a
  pointwise- or BPR-trained model is common practice.
- Cannot be applied to users with no positives or no negatives (delta is always zero).
