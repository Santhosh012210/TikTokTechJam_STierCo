# KuaiRand-Pure Starter Kit

## Dependencies

Python 3.9+ and NumPy. **Nothing else.** PyTorch, pandas, and scikit-learn are not required.

## Data

Download it from https://kuairand.com using the direct Zenodo link; no registration is required:

```bash
# Run from the Starter Kit directory. Extraction creates ./KuaiRand-Pure/.
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Running the baselines

```bash
python3 baseline.py --model fm
```

`--data_dir` defaults to `./KuaiRand-Pure/data`; specify it explicitly if the data is elsewhere.

`--model` accepts `fm` (official baseline), `pop` (trivial baseline), or `random`
(lower bound and evaluation-code sanity check). The full FM run takes about 40 seconds
on one CPU core.

## Task definition (fixed evaluation protocol; do not modify)

| | |
|---|---|
| Task | **Within-user ranking**: rank only each user's impressions in the evaluation set; no full-corpus retrieval |
| Relevance label | `long_view` (native binary column, 0/1) |
| Metrics | `GAUC` and `nDCG@5`; **primary score = their mean** |
| Data splits | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Users with no positives | nDCG is 0.0 and remains in the mean; GAUC includes only users with `0 < positives < impressions`, weighted by positive count |
| nDCG gain | `2^rel − 1` (equivalent to identity for binary labels) |

See `evaluate.py` for the implementation; every convention is documented in its file header.

## Baseline ladder

Scores on the test set. **The FM row is the baseline to beat.**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (lower bound and sanity check) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline)** | **0.6610** | **0.5282** | **0.5946** |

### ⚠️ The real metric range: the nDCG@5 ceiling is 0.729, not 1.0

Among the 23,875 users in the test set:

| | Share | Effect on metrics |
|---|---|---|
| All-negative users (none of the user's impressions are `long_view`) | **27.1%** | nDCG is always **0**, regardless of the model; excluded from GAUC |
| All-positive users | **9.2%** | nDCG is always **1**; excluded from GAUC |
| Discriminative users | **63.7%** | The effective GAUC sample |

Therefore, even using true labels as predictions (an oracle with perfect ranking) reaches only:

| | random | FM baseline | **oracle ceiling** | Available range captured by FM |
|---|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**Measure progress against the oracle ceiling.** Treating 0.5946 as "far from a perfect
1.0" is misleading: the baseline already captures about 30% of the usable range, leaving
0.27 of headroom rather than 0.41.

Across five random seeds, every FM metric has a standard deviation of **0.0008**. The
convergence rule is therefore **ε = 0.002 (approximately 2.5σ), N = 3**: declare convergence
after three consecutive validation iterations improve the primary score by no more than 0.002.

> Sanity check: if `--model random` does not produce primary ≈ 0.475 (±0.001), the
> evaluation harness is incorrect and should be fixed before continuing.

## Submission format

Use a CSV with a header and one row per evaluation-set row:

```text
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| Field | Description |
|---|---|
| `row_id` | Zero-based contiguous row number matching `data.load()[split]` order. Deterministically, `log_standard_4_08_to_4_21_pure.csv` is read before `log_standard_4_22_to_5_08_pure.csv`, then rows are filtered by date while preserving original file order. |
| `user_id` / `video_id` | Redundant fields used only to validate alignment |
| `score` | Any real-valued model score; only relative order matters. NaN and Inf are forbidden. |

> **Why `row_id` is mandatory:** `(user_id, video_id)` is **not unique** in the evaluation
> set. The test set contains 3.06% duplicate pairs, with as many as 12 occurrences, so the
> pair cannot serve as a primary key.

Generate and validate submissions with:

```bash
python3 submit.py --make  --split test  submission.csv    # Generate an example with the official FM baseline
python3 submit.py --check --split test  submission.csv    # Validate format and alignment
python3 submit.py --score --split valid submission.csv    # Validate and score on the local valid split
```

`--check` rejects an invalid header, incorrect row count, discontinuous `row_id`, misaligned
`user_id`/`video_id`, non-numeric scores, and NaN/Inf. **Run `--check` before submitting.**

## Where to start improving

The ordering below is based on measured experiments, not guesses. Dead ends already tested
by the organizers are marked explicitly so participants do not repeat them.

### Measured: these two directions did not help

| Attempt | Result |
|---|---|
| **Add static features**: connect all 13 CWM feature fields (`music_id`, `video_type`, `upload_type`, plus six coarse user-side buckets) | primary **0.5940** versus **0.5950** with five fields: indistinguishable within noise and slightly lower |
| **Increase model capacity**: embedding dimension k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887: effectively flat |

Reason: the `user_id × video_id` interaction already captures most learnable signal. Coarse
buckets such as `follow_user_num_range` are redundant beside `user_id`, and 1.14 million rows
do not support substantially greater capacity. **Features and capacity are not the bottleneck.**

⚠️ Also note: **first-order terms from purely user-side features always contribute zero to
the ranking score.** Ranking happens within each user, so any term constant within a user
cannot change the within-group order. Measured results for `item_pop × user bias` and pure
`item_pop` were identical to every displayed digit. User-side features can help only through
**interactions with item-side features**.

### Unexplored: likely sources of headroom

Ordered by our estimate of their potential. **The organizers have not tested these directions;
they are intentionally left for participants:**

1. **Change the loss function.** The current objective is pointwise log loss, while GAUC and
   nDCG are **ranking metrics**. Pairwise BPR or listwise softmax over each user's impressions
   aligns the objective with evaluation and is the direction we consider most promising.
2. **User-history sequences.** Existing features **do not use behavioral sequences at all**.
   KuaiRand contains hundreds to thousands of training interactions per user, leaving DIN/SIM-
   style interest modeling entirely unexplored.
3. **Multiple objectives.** The logs also contain `is_click`, `is_like`, `is_follow`,
   `is_comment`, `is_forward`, and `play_time_ms`, which can provide auxiliary tasks for the
   primary `long_view` objective.
4. **Watch-time modeling.** [CWM](https://github.com/hyz20/CWM) focuses on **censored
   regression**: true watch time is truncated when a video finishes, motivating a one-sided
   loss instead of squared error. This is a research-rich direction.
5. **Change the model.** Try DeepFM, DCN, or xDeepFM. Because measured capacity was not the
   bottleneck, prioritize directions 1–4 first.
6. **Time features and distribution shift.** Consider `hourmin`, `date`, and shift between
   train and test periods.
7. **Unbiased validation (advanced).** `log_random_4_22_to_5_08_pure.csv` is a randomized-
   exposure log with 1.18 million rows. It can serve as an extra unbiased validation set to
   test whether the model overfits biased traffic.

## Using your own model, including CWM

`evaluate.py` is completely decoupled from the model and needs only three equal-length arrays:

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores))   # scores may come from any model
```

- `user_ids`: the `user_id` for every evaluation-set row
- `labels`: the row's binary `long_view` value
- `scores`: any real-valued score assigned by your model; only relative order matters

You may replace `baseline.py` entirely with PyTorch, LightGBM, or the
[CWM](https://github.com/hyz20/CWM) xDeepFM implementation. Pass the resulting `scores` to
`evaluate()`. **`evaluate.py` is the sole authority for scoring conventions.**

> CWM requires `torch==1.6.0` from 2020, which is unlikely to install on newer GPUs. Its
> loss optimizes counterfactual watch time, while its evaluation label is a reconstructed
> `long_view2`. It is research code for watch-time debiasing and is useful as an advanced
> reference, but it is not recommended as a starting point.

## Files

| | |
|---|---|
| `evaluate.py` | Metric implementation and every evaluation convention. **Do not modify.** |
| `data.py` | Data loading, official splits, and feature encoding. Modify this file to add features. |
| `baseline.py` | Three baselines; FM is the one to beat. |
| `baseline_scores.json` | Official published scores, seed variance, and convergence parameters. |
| `submit.py` | Submission generation and validation. |
| `ablation_features.py` | Feature-ablation experiment reproducing the finding that added static features did not help. |
