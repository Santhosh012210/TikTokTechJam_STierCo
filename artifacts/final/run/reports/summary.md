# Single-agent research run 20260901_012832

## Outcome

- Stop reason: converged
- Converged: true
- Starter Kit task definition confirmed: true
- Reproduced FM baseline validation primary: 0.601469
- Best validation primary: 0.604097
- Best conservative primary: 0.604095
- Delta versus published validation baseline: +0.002497
- Published hidden-test baseline primary: 0.594600
- Hidden-test win status: unmeasured (no hidden labels are exposed to the agent)
- Best frozen frontier node: node_007
- Attempted autonomous experiments: 9
- Successful autonomous experiments: 8
- Failed autonomous experiments: 1

## Validation metrics

| Metric | Official baseline | Best validation | Delta |
|---|---:|---:|---:|
| GAUC | 0.667400 | 0.670621 | +0.003221 |
| nDCG@5 | 0.535700 | 0.537574 | +0.001874 |
| primary | 0.601600 | 0.604097 | +0.002497 |

## Multi-seed stability

Winning candidate `node_007` re-scored on fixed seeds [0, 1, 2, 3, 4] (seed 0 is the sole convergence observation).

| Seed | GAUC | nDCG@5 | primary |
|---:|---:|---:|---:|
| 0 | 0.670621 | 0.537574 | 0.604097 |
| 1 | 0.671137 | 0.538032 | 0.604585 |
| 2 | 0.670642 | 0.537896 | 0.604269 |
| 3 | 0.670599 | 0.537541 | 0.604070 |
| 4 | 0.671144 | 0.537745 | 0.604444 |

primary mean **0.604293** ± std **0.000198** (0.2x the published FM 5-seed std of 0.0008).


## Experiment trajectory

| Iteration | Status | Candidate primary | Incumbent primary | New best |
|---:|---|---:|---:|---|
| 0 | success | 0.601469 | 0.601469 | yes |
| 1 | success | 0.599514 | 0.601469 | no |
| 2 | success | 0.600510 | 0.601469 | no |
| 3 | success | 0.603886 | 0.603886 | yes |
| 4 | success | 0.602510 | 0.603886 | no |
| 5 | success | 0.603620 | 0.603886 | no |
| 6 | success | 0.604097 | 0.604097 | yes |
| 7 | success | 0.603957 | 0.604097 | no |
| 8 | success | 0.602994 | 0.604097 | no |

## Resource usage

- Input tokens: 2836062
- Output tokens: 29604
- GPU-hours: 0.000000
- Total wall seconds: 1256.07
- Manual interventions: 0
- LLM responses: 64
- Tool results: 67
- Provider errors: 0
- Quota pauses: 0
- Maximum automatic quota resumes per invocation: 3
- Maximum wait per quota pause: 300s
- Detailed LLM trace: `/Users/sabithajayakumar/TikTokTechJam_STierCo/artifacts/runs/20260901_012832/logs/llm_events.jsonl`
- Dedicated run Python: `/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/.venv/bin/python`
- Resolved dependency lock: `/Users/sabithajayakumar/TikTokTechJam_STierCo/artifacts/runs/20260901_012832/environment/requirements.lock.txt`
- Resolved distributions: 105

## Prompt templates

```json
[
  {
    "name": "agent.md",
    "path": "mle_agent/research_agent/prompts/agent.md",
    "template_sha256": "336416d0893654423ec83467d163fe413434024cb540b937f696072626b41bfc"
  },
  {
    "name": "prior_findings.md",
    "path": "mle_agent/research_agent/prompts/prior_findings.md",
    "template_sha256": "b09d2cb37527a529e902535805597b5b32b67781ea2415a8f437b49f17d14e30"
  },
  {
    "name": "bootstrap.md",
    "path": "mle_agent/research_agent/prompts/bootstrap.md",
    "template_sha256": "994732e4cf1086b08e8fba50d654f5ff9a091c8f4b0bf7a5b1332d1ef6eaa5e2"
  },
  {
    "name": "iteration.md",
    "path": "mle_agent/research_agent/prompts/iteration.md",
    "template_sha256": "dff84242b4769a34a2196a3a74462720a019e7dca4687a014e8f4725d1a1c964"
  }
]
```

## Retained task context bootstrap

```json
{
  "required": true,
  "complete": true,
  "missing_requirements": [],
  "discovery_completed": true,
  "discovered_documents": [
    {
      "path": "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/README.md",
      "relative_path": "README.md",
      "kind": "task_readme",
      "chars": 9289,
      "lines": 191
    },
    {
      "path": "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/baseline.py",
      "relative_path": "baseline.py",
      "kind": "benchmark_support",
      "chars": 5425,
      "lines": 118
    },
    {
      "path": "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/evaluate.py",
      "relative_path": "evaluate.py",
      "kind": "benchmark_support",
      "chars": 2615,
      "lines": 62
    },
    {
      "path": "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/data.py",
      "relative_path": "data.py",
      "kind": "benchmark_support",
      "chars": 2666,
      "lines": 64
    },
    {
      "path": "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/ablation_features.py",
      "relative_path": "ablation_features.py",
      "kind": "benchmark_support",
      "chars": 3791,
      "lines": 78
    },
    {
      "path": "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/baseline_scores.json",
      "relative_path": "baseline_scores.json",
      "kind": "benchmark_support",
      "chars": 2349,
      "lines": 98
    }
  ],
  "primary_readme_path": "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/README.md",
  "required_evaluation_path": "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/evaluate.py",
  "required_baseline_path": "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/baseline.py",
  "required_data_path": "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/data.py",
  "required_feature_ablation_path": "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/ablation_features.py",
  "required_candidate_model_path": "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_000/model.py",
  "fully_read_paths": [
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/README.md",
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/ablation_features.py",
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/baseline.py",
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/data.py",
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/evaluate.py",
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_000/model.py",
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_001/model.py",
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_002/model.py",
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_003/model.py",
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_004/model.py",
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_005/model.py",
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_006/model.py",
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_007/model.py",
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_008/model.py"
  ],
  "read_coverage": {
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/README.md": [
      [
        0,
        9289
      ]
    ],
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/ablation_features.py": [
      [
        0,
        3791
      ]
    ],
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/baseline.py": [
      [
        0,
        5425
      ]
    ],
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/data.py": [
      [
        0,
        2666
      ]
    ],
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/evaluate.py": [
      [
        0,
        2615
      ]
    ],
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_000/model.py": [
      [
        0,
        5947
      ]
    ],
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_001/model.py": [
      [
        0,
        5947
      ]
    ],
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_002/model.py": [
      [
        0,
        5947
      ]
    ],
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_003/model.py": [
      [
        0,
        5947
      ]
    ],
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_004/model.py": [
      [
        0,
        5947
      ]
    ],
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_005/model.py": [
      [
        0,
        5488
      ]
    ],
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_006/model.py": [
      [
        0,
        5488
      ]
    ],
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_007/model.py": [
      [
        0,
        6792
      ]
    ],
    "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_008/model.py": [
      [
        0,
        8168
      ]
    ]
  },
  "data_inspected": true,
  "environment_inspected": true,
  "environment_inventory": {
    "success": true,
    "python_executable": "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/.venv/bin/python",
    "python_version": "3.13.13",
    "platform": "macOS-26.6.2-arm64-i386-64bit-Mach-O",
    "packages": {
      "catboost": {
        "installed": false,
        "version": null
      },
      "imbalanced-learn": {
        "installed": false,
        "version": null
      },
      "lightgbm": {
        "installed": false,
        "version": null
      },
      "numpy": {
        "installed": true,
        "version": "2.5.2"
      },
      "optuna": {
        "installed": false,
        "version": null
      },
      "pandas": {
        "installed": true,
        "version": "3.0.5"
      },
      "polars": {
        "installed": false,
        "version": null
      },
      "pyarrow": {
        "installed": false,
        "version": null
      },
      "recbole": {
        "installed": false,
        "version": null
      },
      "scikit-learn": {
        "installed": true,
        "version": "1.9.0"
      },
      "scipy": {
        "installed": true,
        "version": "1.18.1"
      },
      "statsmodels": {
        "installed": false,
        "version": null
      },
      "torch": {
        "installed": true,
        "version": "2.13.0"
      },
      "torchaudio": {
        "installed": false,
        "version": null
      },
      "torchvision": {
        "installed": false,
        "version": null
      },
      "torchrec": {
        "installed": false,
        "version": null
      },
      "transformers": {
        "installed": false,
        "version": null
      },
      "xgboost": {
        "installed": false,
        "version": null
      },
      "tensorflow": {
        "installed": false,
        "version": null
      }
    },
    "policy": "The hackathon permits open-source frameworks. Missing allowlisted ML packages auto-install as binary wheels in this run's dedicated venv; off-allowlist packages require explicit approval. Every resolution is logged and frozen.",
    "environment_dir": "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/.venv",
    "auto_install_allowlist": [
      "catboost",
      "imbalanced-learn",
      "lightgbm",
      "numpy",
      "optuna",
      "pandas",
      "polars",
      "pyarrow",
      "recbole",
      "scikit-learn",
      "scipy",
      "statsmodels",
      "torch",
      "torchaudio",
      "torchrec",
      "torchvision",
      "transformers",
      "xgboost"
    ],
    "binary_only_installs": true
  },
  "literature_queries": [
    "ranking loss BPR listwise softmax recommendation",
    "user behaviour sequence modelling DIN SIM click-through rate",
    "multi-task learning auxiliary objectives recommender",
    "within-user recommendation sampled softmax listwise ranking loss hard negative nDCG implementation",
    "recommendation train-only smoothed item target encoding Bayesian shrinkage ranking feature leakage control",
    "Deep Cross Network DCN recommendation explicit feature crosses ranking CTR why improve FM",
    "user behavior history positive interactions recency affinity features recommendation ranking DIN SIM implementation",
    "multi task learning click through rate auxiliary click like profile recommender MMoE ranking implementation task weighting",
    "recommender systems temporal context hour of day feature distribution shift ranking time aware recommendation",
    "censored watch time recommendation duration normalized play progress multi task auxiliary ranking",
    "recommender systems temporal context hour item interaction feature distribution shift ranking time aware recommendation"
  ],
  "baseline_reproduced": true,
  "baseline_metrics": {
    "GAUC": 0.6671326321610643,
    "nDCG@5": 0.5358048805448538,
    "primary": 0.601468756352959,
    "users": 22377.0,
    "rows": 124909.0
  },
  "baseline_execution": {
    "success": true,
    "metrics": {
      "GAUC": 0.6671326321610643,
      "nDCG@5": 0.5358048805448538,
      "primary": 0.601468756352959,
      "users": 22377.0,
      "rows": 124909.0
    },
    "expected_metrics": {
      "GAUC": 0.6674,
      "nDCG@5": 0.5357,
      "primary": 0.6016
    },
    "tolerance": 0.002,
    "error": null,
    "wall_seconds": 14.874451875686646,
    "seed": 0,
    "trial_config": {},
    "execution_id": "08bbbc6585aa48bda51235412efe4234",
    "source_sha256": "3098287eeaae98e05049f646c5a330659bb74823b45214e1af45dd71628aeab3",
    "source_snapshot_path": "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_000/.harness/executions/08bbbc6585aa48bda51235412efe4234/model.py",
    "prediction_sha256": "1a3272e512d79db9878537ab0a02418c08ef1db8ca3d5df4cb2d93203fa28382",
    "organizer_reference": {
      "baseline_path": "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/baseline.py",
      "baseline_sha256": "34d7bfc4b545a38a52dabf3084791e7c7dd6b911eca0e713fb707a6379156d27",
      "scores_path": "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/baseline_scores.json",
      "scores_sha256": "9a0c3575b31684dc8ef4d0762068796ff544b225f42d9bcaf8d53573082df1a5",
      "config": {
        "model": "FM",
        "k": 16,
        "lr": 0.001,
        "batch": 8192,
        "max_epochs": 40,
        "patience": 4,
        "fields": [
          "user_id",
          "video_id",
          "author_id",
          "tab",
          "dur_bucket"
        ]
      },
      "published_valid": {
        "GAUC": 0.6674,
        "nDCG@5": 0.5357,
        "primary": 0.6016
      },
      "published_hidden_test": {
        "GAUC": 0.661,
        "nDCG@5": 0.5282,
        "primary": 0.5946
      },
      "published_hidden_test_std_over_5_seeds": {
        "test_GAUC": 0.0008,
        "test_nDCG@5": 0.0008,
        "test_primary": 0.0008
      },
      "comparison_split": "validation",
      "hidden_test_was_not_scored": true
    }
  },
  "task_context": {
    "task_objective": "Within-user ranking of each user's logged evaluation impressions for long-view relevance; improve the fixed primary score, the mean of GAUC and nDCG@5.",
    "target_label": "long_view, the native binary 0/1 relevance label.",
    "metrics": [
      "GAUC (within-user, only users with both classes, weighted by positive count)",
      "nDCG@5 (within-user; all-negative users contribute 0 and all-positive users contribute 1)",
      "primary = mean(GAUC, nDCG@5)"
    ],
    "data_splits": {
      "train": "20220408\u201320220421",
      "validation": "20220422\u201320220428",
      "test": "20220429\u201320220508 (hidden and never accessed during research)"
    },
    "baseline": "Official NumPy five-field FM (user_id, video_id, author_id, tab, dur_bucket), pointwise logistic loss, k=16, lr=0.001, batch=8192, up to 40 epochs/patience 4. Reproduced validation primary=0.6014687564 (GAUC=0.6671326322, nDCG@5=0.5358048805), consistent with the official validation reference within the 0.002 tolerance.",
    "evaluation_protocol": [
      "Train only on the official train partition and evaluate only on the official validation partition.",
      "Rank only impressions belonging to the same user in the evaluation set; no full-corpus retrieval.",
      "Scores need only preserve relative order but must be finite and aligned one-per-validation-row in data.load() order.",
      "GAUC uses discriminative users only and positive-count weighting; nDCG@5 retains zero-positive users with score zero.",
      "Validation labels may be used for baseline-style early stopping but cannot enter features or prediction construction."
    ],
    "hard_constraints": [
      "Never access test rows, dates, or labels during research.",
      "Never modify official starter-kit files or evaluate.py.",
      "Candidate-data CSVs are immutable; joins, encoders, histories, and transformations run self-contained from candidate model.py over --data_dir.",
      "Fit all vocabularies, bins, aggregates, histories, and target-derived statistics using train rows only; provide unknown defaults for cold items/users.",
      "Preserve the prediction/submission helpers and candidate CLI contract; preserve row order because user-item pairs repeat.",
      "Use fixed seeds for every stochastic source and make substantive model.py changes for each experiment.",
      "Convergence/noise threshold is 0.002 primary."
    ],
    "known_dead_ends": [
      "Organizer feature ablation: adding the organizer's 13 static CWM feature fields did not improve the baseline (primary 0.5940 versus 0.5950 with five fields); do not retry all 13 static fields as a standalone experiment.",
      "Organizer capacity ablation: embedding dimension k=8/16/32 was effectively flat around 0.589; capacity alone is not the demonstrated bottleneck.",
      "Purely user-side first-order features contribute zero to within-user ranking because they are constant within a user's candidate set.",
      "Prior corrected same-user one-negative BPR/schedule improved only +0.000690 primary, below threshold; do not repeat that exact sampler and schedule.",
      "Prior causal exposure-only DIN was a valid negative (0.596819 versus 0.604062 feature-FM parent); do not repeat that exposure-only sequence construction.",
      "Prior leave-one-out target-rate/time FM reconstruction was negative; do not repeat that exact construction.",
      "Original BPR implementation was broken by negative-negative pairing and is not evidence against BPR generally; never use a pair builder without positive/negative/distinct-row assertions."
    ],
    "promising_directions": [
      "Ranking-aligned loss materially different from prior simple BPR, such as listwise sampled softmax, hard-negative, or top-weighted pair loss.",
      "Leakage-safe positive-behavior histories, recency/affinity summaries, or SIM-style retrieval rather than exposure-only DIN.",
      "Controlled extension or isolation of the prior improved train-only rate/context feature bundle, whose effects were confounded.",
      "Multi-task long_view prediction with click/like/profile-entry auxiliary heads, extending rather than exactly rerunning the prior MMoE frontier.",
      "Censored watch-time/progress modelling using only train-observed signals and robust candidate-varying transforms.",
      "Time/context and temporal-shift features fitted on train only, with explicit validation-safe defaults.",
      "A faithful PyTorch DeepFM/DCN/xDeepFM replacement if representation interactions are justified beyond embedding dimension alone."
    ],
    "feature_engineering_context": {
      "baseline_fields": [
        "user_id",
        "video_id",
        "author_id",
        "tab",
        "dur_bucket"
      ],
      "measured_dead_ends": [
        "Organizer result: adding all 13 static CWM feature fields did not improve the five-field baseline (primary 0.5940 versus 0.5950).",
        "Embedding dimension k=8/16/32 stayed flat around 0.589, so capacity alone is not the bottleneck.",
        "Purely user-side first-order features cannot affect within-user ranking.",
        "The exact prior leave-one-out rate/time reconstruction and exposure-only DIN are valid negative implementations."
      ],
      "promising_feature_families": [
        "Train-only smoothed item/user long_view rate or count/context buckets, preferably controlled to isolate the previously confounded positive bundle.",
        "Candidate-varying temporal context (hour/date or train-derived temporal reliability) robust to the observed label shift.",
        "Positive behavioral history affinity/recency and candidate-to-history matching, with no future interactions.",
        "Train-only watch-time/progress summaries and censored-watch-time targets.",
        "Candidate-varying interactions of user state with item/context, rather than pure user-only terms."
      ],
      "leakage_controls": [
        "All categorical vocabularies, bucket edges, rate/count aggregates, and target-derived statistics are fitted on train rows only.",
        "For validation scoring, map unseen users/items/categories to explicit train-derived unknown/prior defaults; EDA reports 17 unseen-item validation rows and 1.59% unseen-user rows.",
        "Construct histories causally from prior train interactions only (or prefixes that exclude the scored event and any validation labels); do not use validation outcome columns.",
        "Do not aggregate validation labels, use validation rows to fit transformations, or key predictions by repeated user-item pairs; write in loader row order."
      ],
      "implementation_boundary": "Feature code runs self-contained from candidate model.py over immutable candidate_data inputs (or hashed local candidate helpers), retaining the inherited prediction writers and CLI contract."
    },
    "candidate_contract": [
      "Accept exactly --data_dir, --seed, --prediction-path with optional --trial-config, or --submission-path for finalization.",
      "Declare accepted trial-config keys and reject unknown keys.",
      "Write row_id,user_id,video_id,score with contiguous zero-based row_id and one finite validation score per data.load() validation row.",
      "Seed NumPy, PyTorch/random if used, and any sampler; restore the best validation checkpoint if early stopping.",
      "Keep write_validation_predictions and write_hidden_submission inside model.py; research runs only use --prediction-path."
    ],
    "source_paths": [
      "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/README.md",
      "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/baseline.py",
      "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/evaluate.py",
      "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/data.py",
      "/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit/ablation_features.py",
      "/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/trial_000/model.py"
    ]
  },
  "research_backlog": [
    {
      "hypothesis": "A controlled extension of the prior pointwise-BCE MMoE frontier that adds one train-only candidate-varying rate/context family at a time (starting with smoothed item long_view-rate buckets plus calendar context) can identify and retain the useful portion of the previously confounded 0.604062 feature gain while keeping auxiliary heads fixed.",
      "target_component": "features",
      "evidence_id": "prior-run:20260831_134234",
      "expected_primary_delta": 0.0025,
      "estimated_cost": "normal; deterministic train-only aggregate construction plus bounded PyTorch training",
      "falsification_criterion": "Primary fails to exceed its matched MMoE/five-field control by 0.0020, or an ablation shows no candidate-varying feature family accounts for the earlier gain."
    },
    {
      "hypothesis": "A listwise within-user sampled-softmax or top-weighted pair loss, with explicit eligible-user grouping and hard negatives, will align training more closely to GAUC/nDCG@5 than pointwise BCE and avoid the prior simple one-negative BPR limitation.",
      "target_component": "loss",
      "evidence_id": "prior-run:20260831_161310",
      "expected_primary_delta": 0.003,
      "estimated_cost": "substantial; custom grouped PyTorch batches, assertions, and early stopping",
      "falsification_criterion": "A valid implementation with positive/negative/group assertions does not improve primary by at least 0.0020 over the same-feature pointwise control."
    },
    {
      "hypothesis": "Compact train-only positive-behavior affinity and recency summaries (clicked/long-viewed author, video, and tag affinities) will outperform the prior exposure-only DIN because they encode preference signal rather than undifferentiated exposure.",
      "target_component": "sequence",
      "evidence_id": "prior-run:20260831_134234",
      "expected_primary_delta": 0.003,
      "estimated_cost": "substantial; causal history build and PyTorch/FM integration",
      "falsification_criterion": "The leakage-safe positive-history representation is valid but is at least 0.0020 below or no better than its matched non-history feature control."
    },
    {
      "hypothesis": "An MMoE extension with calibrated auxiliary weighting or an additional watch-progress-related auxiliary target will transfer abundant click signal without allowing auxiliary labels into validation features, improving the long_view head beyond the prior near-frontier MMoE.",
      "target_component": "auxiliary-task",
      "evidence_id": "prior-run:20260831_161310",
      "expected_primary_delta": 0.0025,
      "estimated_cost": "substantial; seeded multi-head PyTorch with long_view-only validation ranking",
      "falsification_criterion": "Best-checkpoint long_view primary does not exceed the existing MMoE result by 0.0020 under a changed auxiliary weighting/task construction."
    },
    {
      "hypothesis": "Train-only censored watch-time/progress modelling using duration-normalized play-time bins and an auxiliary regression/classification head will supply graded engagement information that binary long_view omits.",
      "target_component": "auxiliary-task",
      "evidence_id": "eda:label-shift",
      "expected_primary_delta": 0.003,
      "estimated_cost": "substantial; robust censor-aware target transforms and multi-task training",
      "falsification_criterion": "After excluding all validation outcomes from feature fitting, the watch-time auxiliary model does not beat an otherwise identical long_view-only model by 0.0020."
    },
    {
      "hypothesis": "A DCN/DeepFM with explicit candidate-varying train-only aggregate/context fields can learn higher-order crosses that the linear-plus-pairwise NumPy FM misses; this is a representation change, not an embedding-dimension-only retry.",
      "target_component": "model",
      "evidence_id": "prior-run:20260831_134234",
      "expected_primary_delta": 0.003,
      "estimated_cost": "substantial; seeded PyTorch model, bounded epochs, and checkpoint restoration",
      "falsification_criterion": "The faithful neural interaction model fails to improve primary by 0.0020 versus the matched FM feature/control model, despite stable learning and early stopping."
    },
    {
      "hypothesis": "Train-fitted temporal reliability and hour/day-context interactions can mitigate the observed train-to-validation long_view-rate shift (0.3366 to 0.3133) and tab-distribution drift while remaining candidate-varying within a user session.",
      "target_component": "features",
      "evidence_id": "eda:label-shift",
      "expected_primary_delta": 0.0025,
      "estimated_cost": "normal; train-only time bucketing/aggregate pipeline and controlled ablation",
      "falsification_criterion": "Temporal/context features do not improve primary by 0.0020 over a matched train-only rate/context control or degrade both GAUC and nDCG@5."
    },
    {
      "hypothesis": "A deployment-robust evaluation/training selection scheme that validates epoch selection on the fixed official primary with deterministic seed controls and explicitly handles unseen-item/user defaults will reduce overfit from temporal drift and stabilize an otherwise promising model.",
      "target_component": "evaluation",
      "evidence_id": "eda:item-cold-start",
      "expected_primary_delta": 0.002,
      "estimated_cost": "normal; checkpoint and cold-start-path audit around a matched candidate",
      "falsification_criterion": "Deterministic checkpoint restoration and explicit unknown paths do not improve or stabilize primary relative to the same model's existing selection protocol."
    }
  ],
  "backlog_reject_count": 0,
  "scored_experiments": [
    {
      "iteration": 1,
      "target_component": "loss"
    },
    {
      "iteration": 2,
      "target_component": "features"
    },
    {
      "iteration": 3,
      "target_component": "model"
    },
    {
      "iteration": 4,
      "target_component": "sequence"
    },
    {
      "iteration": 5,
      "target_component": "auxiliary-task"
    },
    {
      "iteration": 6,
      "target_component": "features"
    },
    {
      "iteration": 7,
      "target_component": "auxiliary-task"
    },
    {
      "iteration": 8,
      "target_component": "features"
    }
  ],
  "rejected_actions": [
    {
      "action": "record_task_context",
      "error": "TASK_CONTEXT_INVALID",
      "invalid_or_empty_fields": [],
      "feature_context_errors": [
        "measured_dead_ends must record that the organizer's 13 static fields did not improve the baseline"
      ],
      "source_errors": [],
      "required_sources_not_cited": [],
      "cited_sources_not_fully_read": []
    }
  ],
  "successful_candidate_fingerprints": 8
}
```

## Architecture

The deterministic Python loop owns an immutable candidate tree. Early experiments branch
from the baseline, later experiments choose among the conservative top frontier with a
noise-scaled UCB score, and rewards/visits propagate through each candidate's lineage.
Every scored node freezes the exact executed source, trial configuration, seed, metrics,
and parent. Final selection is the strongest conservative frozen node, never the latest
working file. LangChain supplies the model adapter and structured tool calls; the retained
Python harness enforces budgets, validation-only execution, baseline verification, and
evidence logging.
