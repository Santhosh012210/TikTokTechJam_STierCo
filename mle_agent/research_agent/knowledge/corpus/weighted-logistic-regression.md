# Weighted logistic regression for watch time

Tags: watch-time, loss, industry

Source: Covington, Adams, Sargin, "Deep Neural Networks for YouTube Recommendations",
RecSys 2016.

## What it is

A trick for making a classifier predict expected watch time without doing regression.
Train logistic regression where positive examples are weighted by their observed watch
time and negatives get weight 1. The learned odds then approximate expected watch time:

    odds = sum(watch_times) / (N - k) ≈ E[watch_time]

for `N` examples with `k` positives, when the click rate is small. Serving uses `exp(logit)`
as the score.

## Why it helps a ranking metric

It gives a watch-time-aware score while keeping the numerically well-behaved logistic
loss, which is far easier to train than raw regression on a heavy-tailed target. Watch
time distributions have extreme outliers, and squared error on them is dominated by the
tail.

For a ranking metric on a watch-time-derived label, a score that estimates expected watch
time is closely aligned with the target while avoiding the outlier sensitivity of direct
regression.

## How to implement

Two lines on top of any logistic model — multiply the per-example loss and gradient by a
weight:

    w = np.where(y == 1, watch_time, 1.0)
    grad = w[:, None] * base_gradient

Clip or log-transform the weights. A single 10-minute view among 30-second videos
produces a weight 20x the typical positive and will dominate a batch. `w = log1p(watch_time)`
or clipping at a high percentile both work; the choice is worth an ablation.

Watch for interaction with duration bias — weighting by raw watch time makes long videos
more influential, which is the opposite of the correction censored regression applies.
The two methods pull in different directions, so combining them naively is not sound.
Decide which framing the data supports before layering both.

## When it will not help

- Without a watch-time column there is nothing to weight by.
- If the scored label is a binary flag with no watch-time relationship, this is
  optimising a different objective than the one being measured.
- The odds-approximation derivation assumes a low positive rate. At high positive rates
  the approximation degrades, though the weighting can still be a useful auxiliary signal.
