"""Deterministic FM root candidate used before autonomous experiments."""
from __future__ import annotations

from harness.config import Config


def make_root_model_py(config: Config) -> str:
    return f'''"""Root candidate: FM baseline adapted for the candidate contract."""
import sys
sys.path.insert(0, r'{config.BASELINE_ROOT}')

import argparse, json
import numpy as np
from data import load, encode
from evaluate import evaluate


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

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', default=r'{config.DATA_DIR}')
a = ap.parse_args()

splits = load(a.data_dir)
if len(splits['train']) != 1_141_112 or len(splits['valid']) != 124_909:
    raise RuntimeError({{"train": len(splits['train']), "valid": len(splits['valid'])}})

enc, dim = encode(splits)
Xtr, ytr, _ = enc['train']
Xva, yva, uva = enc['valid']

m = FM(dim)
rng = np.random.default_rng({config.SEED})
best, best_state, bad = -1, None, 0
for ep in range(40):
    idx = rng.permutation(len(ytr))
    for i in range(0, len(idx), 8192):
        m.step(Xtr[idx[i:i + 8192]], ytr[idx[i:i + 8192]])
    va = evaluate(uva, yva, m.predict(Xva))
    if va['primary'] > best + 1e-5:
        best, bad = va['primary'], 0
        best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
    else:
        bad += 1
        if bad >= 4:
            break

m.V, m.W, m.b = best_state
result = evaluate(uva, yva, m.predict(Xva))
print(json.dumps({{key: float(value) for key, value in result.items()}}))
'''


__all__ = ["make_root_model_py"]
