# MMoE: Multi-gate Mixture-of-Experts

Tags: multi-task, architecture, seesaw

Source: Ma et al., "Modeling Task Relationships in Multi-task Learning with Multi-gate
Mixture-of-Experts", KDD 2018.

## What it is

A multi-task architecture that replaces one shared bottom layer with several parallel
"expert" networks plus a per-task gating network that decides how to mix them:

    expert_outputs = [f_1(x), ..., f_m(x)]        # m shared experts
    g_k = softmax(W_k @ x)                        # one gate per task k
    tower_input_k = sum_i g_k[i] * expert_outputs[i]

Every task sees every expert, but each task learns its own mixing weights. Tasks that
want similar features learn similar gates and share; tasks that conflict learn to
concentrate on different experts and stop interfering.

## Why it helps a ranking metric

It addresses the failure mode of naive multi-task learning: the seesaw effect, where
adding an auxiliary task improves it and degrades the main one because both are forced
through one shared representation that cannot serve both.

Hard parameter sharing (one shared bottom, several heads) works when tasks are closely
related and fails when they are not. In a short-video log, "did the user click" and "did
the user watch to completion" are related but not identical — clickbait scores high on
one and low on the other. That tension is exactly what MMoE's gates are for.

## How to implement

Be realistic about cost. MMoE is a neural architecture: `m` expert MLPs plus `k` gating
networks, all needing forward and backward passes. On top of a linear or FM base model in
pure numpy, this is a substantial rewrite, not an increment.

Order of operations that avoids wasted effort:

1. Implement plain auxiliary-loss multi-task first (see the ESMM note).
2. Measure whether the seesaw actually appears — does the main metric degrade as
   auxiliary weight rises?
3. Only if it does is MMoE addressing a problem you have.

Step 2 matters. MMoE solves a specific pathology; if that pathology is absent, MMoE adds
parameters and training time for nothing, and the extra capacity may not even be usable
on a dataset of this size.

A minimal version: 3-4 experts as single hidden-layer MLPs over the concatenated
embeddings, gates as a single linear layer plus softmax over experts. Keep the expert
count low — the gate has to learn to distinguish them from limited data.

## When it will not help

- When tasks do not conflict. Then hard sharing is already optimal and the gates learn a
  near-uniform mixture, reproducing the shared bottom with extra parameters.
- On small datasets. MMoE multiplies parameter count by roughly the expert count; if
  capacity ablations already showed the model is not underfitting, more capacity is not
  the bottleneck.
- As a first multi-task attempt. Auxiliary losses give most of the benefit for a fraction
  of the code.
