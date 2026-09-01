# Curated prior research findings

This is the agent-facing research memory for the proof of concept. It is reviewed
knowledge, not a dump of the operational experiment log. Use it before ranking the
backlog or proposing an experiment.

The convergence threshold is 0.002 primary. A positive numerical delta smaller than
that is listed under improvement because the score moved upward, but it is not treated
as proof that the hypothesis is supported. Never generalize an implementation failure
into a rejection of its whole research family.

## Things that showed improvement

### Train-only rate and context feature FM

- Evidence: run `20260831_134234`, feature component.
- Result: seed-42 primary `0.604062` versus reproduced baseline `0.601854`
  (`+0.002208`); three-seed mean `0.603562`, std `0.000364`.
- Tested implementation: pointwise-BCE FM with smoothed train-only item and user
  `long_view` rate buckets, calendar-day context, and train user-count buckets.
- Interpretation: the bundle cleared the decision threshold and is useful prior
  evidence, but it did not isolate which field caused the gain.
- Research action: extend or ablate this family; do not spend an experiment merely
  recreating the exact bundle.
- Candidate fingerprint: `0ead8d497d31f68c539098f5d2ca91fefbcc33cc2e3c887766a4616df4070ce5`.

### Corrected same-user BPR

- Evidence: run `20260831_161310`, loss component.
- Result: primary `0.602544` versus pointwise five-field parent `0.601854`
  (`+0.000690`); GAUC improved while nDCG@5 moved slightly down.
- Tested implementation: one deterministic same-user negative per eligible positive,
  with assertions that positive and negative labels and row indices are valid.
- Interpretation: a valid run and a small numerical improvement, but below the 0.002
  threshold; the hypothesis was recorded as `not_supported`.
- Research action: do not repeat this exact sampler and schedule. A materially different
  loss test such as listwise softmax, hard-negative sampling, or top-weighted pairs is
  still open.
- Candidate fingerprint: `5e496ec6b0e50c1416a917343bff49a62323dde3271ab070ddc94f918e911113`.

### Multi-task MMoE

- Evidence: run `20260831_161310`, auxiliary-task component.
- Result: seed-42 primary `0.603721`; three-seed mean `0.604068`, std `0.000275`.
  This is `+0.001867` over the reproduced five-field baseline at seed 42 and
  approximately matches the earlier `0.603718` development frontier.
- Tested implementation: seeded PyTorch MMoE with three shared experts, task-specific
  gates, and `long_view`, click, like, and profile-entry heads; validation ranking used
  only the long-view head.
- Interpretation: current best measured candidate, but its `+0.001177` gain over its
  BPR parent is below the 0.002 threshold; the hypothesis was recorded as
  `not_supported` rather than a breakthrough.
- Research action: build from or ablate the multi-task mechanism. Do not rerun the exact
  architecture and weights unchanged.
- Candidate fingerprint: `3829ea42dddd5c43b1c0dad671af1874e62535439a1b9edd56b00b54de95aeba`.

## Things that did not show improvement

### All 13 static feature fields

- Result: organizer ablation approximately `0.5940` versus `0.5950` for the five-field
  reference in that ablation.
- Interpretation: adding every available static field is a settled negative.
- Do not retry: static-field expansion as a standalone experiment.
- Still open: candidate-varying crosses, causal histories, temporal signals, watch-time
  signals, and train-fitted aggregates with a specific mechanism.

### Embedding dimension alone

- Result: `k=8`, `k=16`, and `k=32` stayed approximately flat near `0.589` in the
  organizer ablation.
- Interpretation: capacity alone is not the demonstrated bottleneck.
- Do not retry: an experiment whose only substantive change is embedding dimension.

### Pure user-side first-order features

- Result: these terms are constant inside one user's candidate set and therefore cannot
  change within-user ordering.
- Do not retry: pure user-only first-order features as a ranking experiment.
- Still open: interactions between user state and candidate-varying item or context
  features.

### Original broken BPR implementation

- Evidence: run `20260831_134234`, loss component.
- Result: primary `0.499380`, GAUC `0.522289`, no better than a trivial ranker.
- Diagnosis: `avalues()` was unpacked so the variable named `p` came from the negatives
  slot. Training received negative-negative pairs and never supplied a positive example
  to the BPR loss.
- Interpretation: `suspect_implementation`, not evidence against BPR. The corrected BPR
  experiment above is the valid test of the simple sampler.
- Never retry: this code or any pair builder without explicit positive/negative label and
  distinct-row assertions.
- Candidate fingerprint: `3c93886590858630e012e53a1e7adb2a5b4a11713546ac4cd77c2311f39bed96`.

### Causal exposure-only DIN

- Evidence: run `20260831_134234`, sequence component.
- Result: primary `0.596819` versus feature-FM parent `0.604062` (`-0.007243`).
- Interpretation: the leakage-safe exposure-prefix implementation was a valid negative
  result; unlabelled exposure history alone did not beat the static train-derived
  representation.
- Do not retry: the same exposure-only DIN construction.
- Still open: materially different positive-behaviour histories, compact affinity or
  recency summaries, SIM-style retrieval, and sequence objectives with richer signals.
- Candidate fingerprint: `89febd57278db2b3211dd9ad41c5425a3d236122075e8ffdf945f901a1fd3f1f`.

### Leave-one-out rate/time feature reconstruction

- Evidence: run `20260831_161310`, feature component.
- Result: primary `0.600522` versus five-field parent `0.601854` (`-0.001332`).
- Tested implementation: leave-one-out train item/user target-rate buckets, frozen hour
  context, and train-user-count bucket under the original pointwise FM.
- Interpretation: a valid negative result for this reconstruction. It does not erase the
  earlier feature bundle's improvement because the feature construction differed.
- Do not retry: this exact leave-one-out reconstruction.
- Candidate fingerprint: `1e84bc1c939dbb7ece47224c90c335594a7789602ef1a1fa41e31450049e291e`.

## How to use these findings

1. Do not propose an exact implementation or fingerprint listed above.
2. Treat “do not retry” as scoped to the described implementation, unless the finding
   explicitly establishes a mathematical or organizer-measured dead end.
3. A retry of an open family must state the new mechanism and why it avoids the recorded
   limitation.
4. Prefer experiments that isolate or extend a positive mechanism over experiments that
   merely reconstruct an old score.
5. The harness's operational history remains authoritative for exact fingerprints and
   measured metrics, but this reviewed file is authoritative for research interpretation.
