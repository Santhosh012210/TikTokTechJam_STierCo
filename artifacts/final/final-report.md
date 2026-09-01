# Final submission

- Source autonomous run: `20260901_012832`
- Source frozen node: `node_010`
- Research frontier best node: `node_007`
- Stop reason: `converged`
- Converged: `true`
- Manual interventions during autonomous research: `0`
- Input/output tokens: `2836062` / `29604`
- GPU-hours: `0.0`
- Python environment: `/Users/sabithajayakumar/TikTokTechJam_STierCo/experiment_workspace/20260901_012832/.venv/bin/python`
- Resolved dependency lock: `/Users/sabithajayakumar/TikTokTechJam_STierCo/artifacts/final/requirements.lock.txt`
- Per-iteration run logs: `/Users/sabithajayakumar/TikTokTechJam_STierCo/artifacts/final/run`
- Published organizer hidden-test baseline primary: `0.594600`
- Hidden-test win status: `unmeasured`; organizer evaluation is required
- Multi-seed primary: mean `0.604293` ± std `0.000198` (published FM 5-seed std `0.0008`); seed 0: 0.604097; seed 1: 0.604585; seed 2: 0.604269; seed 3: 0.604070; seed 4: 0.604444

- Submission compatibility fallback: not required

- Submission-contract repair: selected frozen child of `node_007`; the child reproduced all validation seeds before promotion

| Validation metric | Official baseline | Final candidate | Delta |
|---|---:|---:|---:|
| GAUC | 0.667400 | 0.670621 | +0.003221 |
| nDCG@5 | 0.535700 | 0.537574 | +0.001874 |
| primary | 0.601600 | 0.604097 | +0.002497 |

Submission validation: `✓ format and alignment validated: 170,588 rows, split=test`

The trusted finalizer generated row-aligned hidden-split predictions but did not compute or expose
hidden-test metrics to the research agent.

## Task definition

The brief's judging criteria mention `click` / NDCG@10 / Recall@50, while the checked-in Starter
Kit (`evaluate.py`, `baseline_scores.json`, its README, and the brief's own line 74) defines the
target as `long_view` scored by GAUC and nDCG@5. This entry was built to the **Starter Kit** as the
executable authority the submission is checked against -- a deliberate reading of an internally
inconsistent brief, not an oversight. `--task-definition-confirmed` records that this reading was
made intentionally.
