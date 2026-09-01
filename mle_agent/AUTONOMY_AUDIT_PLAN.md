# MLE agent autonomy audit and implementation plan

Goal: the official run must autonomously search for a candidate that beats the reproduced
Starter Kit baseline, retain scientifically valid evidence, and promote the best verified
candidate rather than whichever file happened to be edited last.

## Audit findings

1. The official runner was a greedy incumbent chain. A legacy tree existed in the repository,
   but its visits never changed, rewards were not back-propagated, and the official entrypoint did
   not use it.
2. Candidate identity was source-only. Seed and trial configuration could change without changing
   the repeat fingerprint, and a post-run edit could make recorded metrics refer to different code.
3. Selection used one validation observation and convergence watched one incumbent. This encouraged
   validation noise chasing and could terminate before the broader research frontier had matured.
4. The finalizer promoted a source path without restoring its trial configuration or seed and did
   not reproduce the recorded validation score before touching the hidden split.
5. Wall deadline, remaining scored-variant budget, and iteration metadata existed in the tool
   runtime but were not wired through the official agent. A sweep could therefore exceed the
   intended experiment budget and a long job could overrun the wall limit.
6. Documented quota recovery was not implemented in the active loop, cost exhaustion was not a
   distinct stop reason, and the official $1 default was too small for its advertised 50-experiment
   run.
7. Candidate subprocesses lost the obvious LLM keys and hidden rows, but could still read unrelated
   filesystem paths, inherit unknown credentials, launch network clients, or attempt child-process
   escape.
8. Cross-run history gated exact failures but did not preserve the strongest executable candidate.
   A later run always restarted from the baseline source.
9. Provider capability claims exceeded installed LangChain integrations and audited price entries.
   Setup documentation also disagreed with the actual output-token limit and artifact retention.
10. The harness loaded the organizer's validation thresholds, but used seed 42 and did not retain a
    complete hashed statement of the official FM definition. Published hidden-test reference rungs
    also needed to be kept distinct from the validation rungs used during autonomous search.

## Implemented plan

- [x] Add a proper immutable candidate tree. Every frozen node records source hash/path, parent,
  hypothesis/component, trial configuration, seed, metrics, stability, children, visits, and reward.
- [x] Force the first four breadth branches to cover loss, features, model, and sequence from the
  baseline. Afterwards select parents from the conservative top three using noise-scaled UCB plus
  back-propagated branch reward.
- [x] Snapshot source before execution, use unique execution/prediction paths, reject source mutation
  during a run, and fingerprint source + seed + canonical trial configuration.
- [x] Re-score competitive nodes on fixed seeds and rank by mean minus standard deviation. Evaluate
  convergence over the whole top frontier, with a minimum of eight scored variants.
- [x] Freeze each scored source/config/seed bundle. Store the entire frontier in run metrics and make
  the winning frozen node—not the latest trial file—the only finalization input.
- [x] Re-run the frozen winner on the preserved validation view during finalization, require exact
  metric reproduction, and use the identical seed/config for hidden-split submission generation.
- [x] Wire run deadline, iteration, and remaining variant budget into bootstrap and experiment tool
  runtimes. Bound multi-seed checks by the same deadline.
- [x] Implement bounded unattended quota recovery, explicit cost-limit evidence/stop reason, and an
  official-run cost ceiling that can support the advertised search while remaining user-overridable.
- [x] Add a startup audit sandbox, environment allowlist, filesystem allowlist, network/process bans,
  and static bans on common native/network escape modules.
- [x] Archive each winning frozen bundle as a versioned local champion and import the last verified
  champion as a branch in the next run.
- [x] Install every registered LangChain provider integration and fail closed on model IDs without an
  explicit audited token-price entry.
- [x] Pin the organizer FM config and source/score hashes, use its published seed panel 0–4, compare
  all three validation metrics, and label hidden-test performance as unmeasured until organizer
  evaluation. Validation sanity uses the corresponding validation rungs (random 0.4834, popularity
  0.5807), while hidden references remain 0.4753, 0.5715, and FM 0.5946.
- [x] Update setup/architecture documentation and add offline regression tests for candidate identity,
  node immutability, lineage reward updates, and sandbox escape prevention.

## Acceptance checks

- Offline agent, memory, knowledge, schema, recovery, and Starter Kit integrity suites pass.
- A scored execution's source hash equals its frozen node hash.
- Re-freezing a node fails; ancestor visits/rewards change after every descendant result.
- Changing only seed or trial configuration changes candidate identity.
- Filesystem escape and network imports are rejected.
- Final metrics contain the complete frontier, conservative winner, frozen bundle manifest, provider
  cost, and cross-run champion provenance.
