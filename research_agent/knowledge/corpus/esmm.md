# ESMM: Entire Space Multi-Task Model

Tags: multi-task, auxiliary-loss, sample-selection-bias, architecture

Source: Ma et al., "Entire Space Multi-Task Model: An Effective Approach for Estimating
Post-Click Conversion Rate", SIGIR 2018. arXiv:1804.07931

## What it is

A structure for learning a rare downstream signal that is only observed conditional on an
upstream one. The canonical case is click then conversion: conversion is only observed on
clicked items, so a model trained on clicks-only data is trained on a biased subsample and
then applied to all impressions.

ESMM's trick is to never train the conditional model directly. It models two quantities
over the *entire* impression space and multiplies them:

    p(click & engage | impression) = p(click | impression) * p(engage | click)

Both towers share their embedding layer. The loss is applied to `p(click)` and to the
*product* — never to `p(engage | click)` in isolation — so the conditional tower is
learned implicitly, on the full impression space, without ever being fed a biased sample.

## Why it helps a ranking metric

Two distinct mechanisms, and it is worth being clear about which one is doing the work.

The first is the bias correction described above, which matters when the scored label
itself is conditional.

The second, and usually the more relevant one for a logged ranking dataset, is
**shared-embedding regularisation**. Auxiliary signals that correlate with the target
give the shared embedding table far more gradient signal than the target alone. In a
short-video log, signals like click, like, follow, comment, forward, and completion are
all correlated with genuine interest. If the scored label is sparse, training the shared
embeddings against several correlated labels at once produces better item and user
representations than training against one.

This helps a within-user ranking metric because it improves the *item-side* embeddings,
which vary within a user's list.

## How to implement

The cheapest useful version is not full ESMM at all — it is auxiliary-loss multi-task,
which is a few lines:

    total_loss = loss(target_label) + sum_k( w_k * loss(auxiliary_label_k) )

with one output head (a separate weight vector `W_k` and bias) per label, all reading the
same shared embeddings. Backpropagation scatters each head's gradient into the shared
table through the same `np.add.at` path already used for the main loss.

Start with `w_k` around 0.1-0.3. Auxiliary weights that are too high let a dense
auxiliary label dominate the shared representation and pull it away from the target.

The full ESMM product structure only earns its complexity when the target label is
genuinely conditional on another observed event. Check the data first: if the target is
observed on every impression rather than only on clicked ones, there is no sample
selection bias to correct and the product structure adds nothing over auxiliary losses.

Sanity check to run before investing: compute the correlation between each auxiliary
label and the target, and the base rate of each. An auxiliary label that is either
near-zero correlated or vanishingly rare will not help.

## When it will not help

- If the target label is already observed over the entire impression space, the
  bias-correction half of the argument does not apply. The regularisation half still
  might.
- Auxiliary labels much rarer than the target contribute little gradient.
- Too many auxiliary tasks with high weights degrades the main task — the seesaw effect.
  If that happens, that is the signal to move to MMoE or PLE, which exist precisely to
  address it.
