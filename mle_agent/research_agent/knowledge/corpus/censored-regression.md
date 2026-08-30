# Censored regression for watch time

Tags: watch-time, loss, duration-bias, censoring

Source: Zhao et al., "Counteracting Duration Bias in Video Recommendation via
Counterfactual Watch Time", KDD 2024. arXiv:2406.07932. Code: github.com/hyz20/CWM

## What it is

Watch time on a short video is a *censored* observation. If a user watches a 15-second
video for all 15 seconds, the recorded watch time is 15 — but the quantity of interest is
how long they *would* have watched, which is at least 15 and possibly much more. The
video ended; the interest did not.

Squared error treats that 15 as an exact target and penalises the model for predicting
20, which is wrong: 20 is entirely consistent with the observation. Censored regression
uses a one-sided loss instead — for a completed play, the model is penalised only for
predicting *below* the observed duration:

    if play was completed (watch_time >= duration):
        loss = max(0, duration - prediction) ** 2       # penalise under-prediction only
    else:
        loss = (watch_time - prediction) ** 2           # exact observation, normal loss

The KDD 2024 formulation goes further, defining counterfactual watch time — the time a
user would spend if duration were unbounded — and learning it by maximising a
counterfactual likelihood over observed watch times, with a cost-based transform mapping
that quantity to user interest.

## Why it helps a ranking metric

Duration bias is the concrete problem: because short videos are completed far more often
than long ones, raw watch time and completion rate both correlate strongly with video
duration rather than with user interest. A model trained naively on watch time learns to
rank short videos highly, which is a duration artefact, not a preference.

If the scored label is itself derived from watch time (a "long view" or completion flag,
for instance), the label inherits this bias, and correcting for it targets the actual
scoring objective rather than a proxy.

The correction varies *within* a user's impression list because duration varies within it,
so it is capable of reordering — unlike a user-level correction, which would cancel.

## How to implement

Diagnose before implementing. The check is cheap and decides whether this is worth doing:

1. Bucket impressions by video duration.
2. Compute the positive rate of the scored label per bucket.
3. Plot or tabulate. A strong monotone relationship between duration and label rate is
   duration bias, and confirms the premise.

If the relationship is strong, the simplest interventions in increasing order of cost:

**Duration-debiased label.** Instead of the raw label, use the label residualised against
duration — the difference between this row's completion and the average completion for
its duration bucket. Rank on that.

**Watch-time-percentile target.** Convert watch time into its percentile *within the
video's duration bucket*, which removes the duration scale entirely. Train a regression
against the percentile and use it as an auxiliary task alongside the main label.

**One-sided loss.** As written above, requires the completion indicator. In numpy the
gradient is a masked linear term:

    residual = prediction - target
    completed_and_over = completed & (residual > 0)
    grad = 2 * residual
    grad[completed_and_over] = 0.0      # no penalty for over-predicting a completed play

Note the model must still be *scored* on the original label — the debiasing changes what
is trained on, never what is evaluated.

## When it will not help

- If the duration-vs-label-rate check comes back flat, there is no duration bias in this
  data and the whole premise is absent. Run the check first.
- If watch-time and duration columns are not both present, none of this is computable.
- The full counterfactual formulation is a research-grade method whose reference
  implementation targets a different reconstructed label and pins an old framework
  version. Treat the paper as the idea source and implement the one-sided loss directly
  rather than trying to run the original code.
