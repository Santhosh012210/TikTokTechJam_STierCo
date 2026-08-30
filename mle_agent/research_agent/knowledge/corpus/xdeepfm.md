# xDeepFM

Tags: architecture, feature-interaction, ctr

Source: Lian et al., "xDeepFM: Combining Explicit and Implicit Feature Interactions for
Recommender Systems", KDD 2018. arXiv:1803.05170

## What it is

DeepFM with the FM branch replaced by a Compressed Interaction Network (CIN). CIN
computes interactions at the *vector* level rather than the bit level: each layer takes
outer products between the current feature-map matrix and the original embedding matrix,
then compresses the result with learned filters.

    Z^{k} = X^{k-1} outer X^{0}                # (H_{k-1}, m, D) tensor
    X^{k}_h = sum over (i,j) of W^{k,h}_{ij} * Z^{k}_{ij}

The distinction from DCN is that CIN's interactions stay at embedding-vector granularity,
which the authors argue is more natural for categorical embeddings.

## Why it helps a ranking metric

The claimed advantage over DCN is vector-wise rather than bit-wise interaction, and over
the MLP is explicitness. In the paper's benchmarks it edges out both.

The margins reported are small, and the compute cost is the highest of the three
interaction architectures in this corpus — the outer-product tensor is `O(H * m * D)` per
layer.

## How to implement

Honestly: this is the least attractive of the interaction architectures to hand-write.
The outer-product-and-compress structure means the backward pass involves gradients
through a three-dimensional tensor contraction, which is error-prone in numpy.

If the interaction-capacity hypothesis is worth testing at all, DCN-V2 tests it for a
fraction of the implementation risk. Reach for xDeepFM only if DCN-V2 produced a real
gain and the question is whether more of the same helps.

## When it will not help

All of the DeepFM and DCN caveats, amplified by cost:

- Flat capacity ablations are evidence against the whole family.
- Highest implementation and compute cost of the three for the smallest reported margin.
- Model-family changes do not fix an objective misaligned with the metric.
