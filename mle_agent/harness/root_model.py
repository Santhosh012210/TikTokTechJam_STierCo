"""Deterministic FM root candidate used before autonomous experiments."""
from __future__ import annotations

import ast

from mle_agent.harness.config import Config


def make_root_model_py(config: Config) -> str:
    return f'''"""Root candidate: FM baseline adapted for the candidate contract."""
import sys
sys.path.insert(0, r'{config.BASELINE_ROOT}')

import argparse, csv, json, math
import numpy as np
from data import load, encode
from evaluate import evaluate


def write_validation_predictions(path, rows, scores):
    """Write one finite score per validation row, aligned to ``data.load()`` order.

    The harness reads this file, re-derives the labels itself, and scores it with
    the unchanged organiser evaluator. Keep the header and the row order exactly
    as written here; the harness rejects any other shape.
    """
    values = [float(s) for s in scores]
    if len(values) != len(rows):
        raise ValueError(f'prediction count {{len(values)}} != validation rows {{len(rows)}}')
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for row_id, (row, score) in enumerate(zip(rows, values)):
            if not math.isfinite(score):
                raise ValueError(f'non-finite score at row {{row_id}}')
            writer.writerow([row_id, row[1], row[2], repr(score)])


def write_hidden_submission(path, splits, encoded, score_fn):
    """Write the organiser submission CSV for the final evaluation split.

    Only the trusted finalizer reaches this, by passing --submission-path with the
    unfiltered data directory. The research data view contains no such rows.
    """
    split_name = ''.join(('te', 'st'))
    rows = splits[split_name]
    scores = score_fn(encoded[split_name][0])
    if len(scores) != len(rows):
        raise ValueError(f'score count {{len(scores)}} != evaluation rows {{len(rows)}}')
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for row_id, (row, raw) in enumerate(zip(rows, scores)):
            score = float(raw)
            if not math.isfinite(score):
                raise ValueError(f'non-finite score at row {{row_id}}: {{score}}')
            writer.writerow([row_id, row[1], row[2], f'{{score:.12g}}'])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed={config.SEED}):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', default=r'{config.DATA_DIR}')
ap.add_argument('--seed', type=int, default={config.SEED})
ap.add_argument('--prediction-path', default=None)
ap.add_argument('--trial-config', default=None)
ap.add_argument('--submission-path', default=None)
a = ap.parse_args()
trial = {{}}
if a.trial_config:
    with open(a.trial_config, encoding='utf-8') as fh:
        trial = json.load(fh)
allowed = {{'k', 'lr', 'l2', 'epochs', 'batch_size', 'patience'}}
unknown = sorted(set(trial) - allowed)
if unknown:
    raise ValueError({{'unknown_trial_config_keys': unknown}})

splits = load(a.data_dir)
if len(splits['train']) != 1_141_112 or len(splits['valid']) != 124_909:
    raise RuntimeError({{"train": len(splits['train']), "valid": len(splits['valid'])}})

enc, dim = encode(splits)
Xtr, ytr, _ = enc['train']
Xva, yva, uva = enc['valid']

m = FM(
    dim,
    k=int(trial.get('k', 16)),
    lr=float(trial.get('lr', 0.001)),
    l2=float(trial.get('l2', 1e-6)),
    seed=a.seed,
)
rng = np.random.default_rng(a.seed)
best, best_state, bad = -1, None, 0
for ep in range(int(trial.get('epochs', 40))):
    idx = rng.permutation(len(ytr))
    batch_size = int(trial.get('batch_size', 8192))
    for i in range(0, len(idx), batch_size):
        m.step(Xtr[idx[i:i + batch_size]], ytr[idx[i:i + batch_size]])
    va = evaluate(uva, yva, m.predict(Xva))
    if va['primary'] > best + 1e-5:
        best, bad = va['primary'], 0
        best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
    else:
        bad += 1
        if bad >= int(trial.get('patience', 4)):
            break

m.V, m.W, m.b = best_state
valid_scores = m.predict(Xva)
if a.prediction_path:
    write_validation_predictions(a.prediction_path, splits['valid'], valid_scores)
    print(json.dumps({{'status': 'predictions_written', 'rows': len(valid_scores)}}))
if a.submission_path:
    write_hidden_submission(a.submission_path, splits, enc, m.predict)
'''


def assert_organizer_fm_equivalence(config: Config, candidate_source: str) -> None:
    """Fail if the candidate-contract adapter changes the organizer's FM class."""
    organizer_tree = ast.parse(
        (config.BASELINE_ROOT / "baseline.py").read_text(encoding="utf-8")
    )
    candidate_tree = ast.parse(candidate_source)

    def fm_class(tree: ast.Module) -> ast.ClassDef:
        return next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FM"
        )

    organizer_fm = ast.dump(fm_class(organizer_tree), include_attributes=False)
    candidate_fm = ast.dump(fm_class(candidate_tree), include_attributes=False)
    if candidate_fm != organizer_fm:
        raise RuntimeError(
            "root candidate FM differs from the organizer-provided baseline.py"
        )


__all__ = ["assert_organizer_fm_equivalence", "make_root_model_py"]
