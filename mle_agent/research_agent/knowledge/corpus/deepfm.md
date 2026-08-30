# DeepFM

Tags: architecture, feature-interaction, ctr

Source: Guo et al., "DeepFM: A Factorization-Machine based Neural Network for CTR
Prediction", IJCAI 2017. arXiv:1703.04247

## What it is

An FM and a deep MLP in parallel, sharing one embedding table, with their outputs summed:

    y = sigmoid( y_FM + y_DNN )

The FM branch captures explicit second-order interactions; the MLP branch captures
implicit higher-order ones. Shared embeddings mean no separate feature engineering for
the two branches, which was the improvement over Wide and Deep.

## Why it helps a ranking metric

The claim is that FM is limited to pairwise interactions, and some signal is genuinely
higher-order — the joint effect of three or more fields that no pair captures.

Be sceptical of that claim in any specific case. It is dataset-dependent and frequently
does not hold on datasets with few fields and high-cardinality ID features, where a
learned `user_id x item_id` interaction already carries most of the available signal and
higher-order combinations of a handful of categorical fields add little.

## How to implement

The FM branch already exists in any FM baseline. The DNN branch is: concatenate the field
embeddings into one vector, pass through 2-3 hidden layers with ReLU, output a scalar.
Add to the FM logit.

In pure numpy that means writing forward and backward for an MLP — matmuls, ReLU masks,
and the gradient back into the shared embedding table. Tractable, but note the shared
table now receives gradients from two paths and both must accumulate into it.

Typical hidden sizes are [400, 400, 400] in the paper; on a small dataset start far
smaller, [64, 32], or the MLP will overfit before it contributes.

## When it will not help

**Check the capacity evidence first.** If ablations on embedding dimension already showed
the metric is flat across a range of capacities, the model is not underfitting, and
DeepFM's contribution is more capacity. That is direct evidence against this direction.

Other conditions where it disappoints:

- Few feature fields. Higher-order interactions need fields to interact; with a handful
  of categoricals there is not much beyond pairwise to find.
- Datasets around a million rows, which do not support the MLP's parameter count.
- When the objective is misaligned with the metric. Changing the architecture while
  keeping a pointwise loss under a ranking metric optimises the wrong thing harder.
  Fix the objective before the architecture.
