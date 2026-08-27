# Within-user ranking: what can and cannot change the score

Tags: evaluation, ranking-objective, task-property, gauc, ndcg

Source: Property of group-wise ranking metrics. GAUC: Zhou et al., KDD 2018 (introduced
alongside DIN). nDCG: Järvelin and Kekäläinen, TOIS 2002.

## What it is

A structural constraint that applies whenever the metric ranks items *inside* a group
rather than across a whole catalogue. Group-wise metrics — GAUC, per-user nDCG — compute
a score inside each user's candidate list and then average over users.

The consequence: **any term that is constant within a group cannot change that group's
ordering.** Adding a constant to every score in one user's list leaves the order
identical, so it leaves that user's GAUC and nDCG identical.

## Why it helps a ranking metric

Knowing this in advance eliminates entire categories of change that cannot possibly work,
which is worth more than most positive ideas.

Things that are constant within a user, and therefore contribute exactly zero:

- A user bias term.
- Any first-order term over a pure user-side feature — user demographics, activity
  buckets, follower counts, tenure.
- A user embedding used only in a first-order (non-interacting) position.
- Any global constant or calibration shift.

Things that vary within a user, and therefore can contribute:

- Item-side terms of any order — item bias, item popularity, item features.
- **Crosses** between a user-side feature and an item-side feature. This is how user
  information becomes usable: `user_segment x item_category` varies across the items in
  one user's list even though `user_segment` alone does not.
- Anything that depends on the candidate item, including candidate-dependent user
  representations (this is the core mechanism behind attention-based interest models).

## How to implement

Use it as a screening test before writing code. For any proposed feature or term, ask:
does its value differ between two rows belonging to the same user? If no, it cannot move
the metric and the iteration is wasted.

Two practical consequences:

**Adding user-side features requires crossing them.** A user feature added as a
standalone field contributes only a first-order term that cancels. It must interact with
an item-side field to have any effect. In an FM this happens automatically through the
second-order term, which is why FM tolerates user features that a linear model would
waste entirely — but the first-order part still contributes nothing.

**Ranking losses encode this constraint automatically.** Pairwise loss depends on
`s_i - s_j` within a user, so user-constant terms cancel in the loss itself and receive
no gradient. Listwise softmax is shift-invariant within a group for the same reason.
Pointwise loss has no such property: it will spend capacity fitting user-constant terms
that the metric then ignores. This is an argument for objective alignment that is
independent of the usual calibration-versus-ordering argument.

A quick empirical confirmation: score a validation split with an item-popularity model,
then again with item popularity plus any user-side bias term. If the metric is identical
to the last digit, the constraint is confirmed for that data.

## When it will not help

- It does not apply to global-ranking or retrieval metrics, where all users' items are
  ranked together and user-side terms do matter.
- It says nothing about whether an item-side or crossed feature *will* help — only that a
  user-constant one cannot. It rules things out, it does not rule things in.
