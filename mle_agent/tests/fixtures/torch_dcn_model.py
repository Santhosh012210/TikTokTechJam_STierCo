"""Reference framework-backed candidate: a Deep & Cross Network in PyTorch.

This is a *contract-faithful example* of a `model.py` an autonomous experiment
could produce when it chooses a framework model. The harness runs candidate code
with only the organiser starter kit on PYTHONPATH, so this file:

- takes exactly the inherited CLI (`--data_dir --seed --prediction-path
  [--trial-config] --submission-path`);
- imports only `data`/`evaluate` from the starter kit, plus torch and numpy;
- writes the aligned validation prediction CSV itself (header + row order fixed);
- keeps `write_hidden_submission` for the trusted finaliser only;
- seeds torch / numpy / random from `--seed`, bounds epochs, early-stops on the
  official validation primary, and restores the best checkpoint;
- never imports `mle_agent` and never launches a subprocess or package manager.

The agent is free to change the architecture; the contract above is what must
survive.
"""
import argparse
import csv
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn

from data import load, encode
from evaluate import evaluate


def write_validation_predictions(path, rows, scores):
    """One finite score per validation row, aligned to ``data.load()`` order."""
    values = [float(s) for s in scores]
    if len(values) != len(rows):
        raise ValueError(f"prediction count {len(values)} != validation rows {len(rows)}")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (row, score) in enumerate(zip(rows, values)):
            if not math.isfinite(score):
                raise ValueError(f"non-finite score at row {row_id}")
            writer.writerow([row_id, row[1], row[2], repr(score)])


def write_hidden_submission(path, splits, encoded, score_fn):
    """Organiser submission CSV for the final split. Trusted finaliser only."""
    split_name = "".join(("te", "st"))
    rows = splits[split_name]
    scores = score_fn(encoded[split_name][0])
    if len(scores) != len(rows):
        raise ValueError(f"score count {len(scores)} != evaluation rows {len(rows)}")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (row, raw) in enumerate(zip(rows, scores)):
            score = float(raw)
            if not math.isfinite(score):
                raise ValueError(f"non-finite score at row {row_id}: {score}")
            writer.writerow([row_id, row[1], row[2], f"{score:.12g}"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class DCN(nn.Module):
    """Deep & Cross Network over the shared field-embedding table from ``encode``."""

    def __init__(self, dim: int, n_fields: int, k: int = 16,
                 cross_layers: int = 3, mlp: tuple[int, ...] = (128, 64)):
        super().__init__()
        self.emb = nn.Embedding(dim, k)
        nn.init.normal_(self.emb.weight, std=0.01)
        in_dim = n_fields * k
        self.cross_w = nn.ParameterList(
            nn.Parameter(torch.zeros(in_dim, 1)) for _ in range(cross_layers)
        )
        self.cross_b = nn.ParameterList(
            nn.Parameter(torch.zeros(in_dim)) for _ in range(cross_layers)
        )
        layers: list[nn.Module] = []
        prev = in_dim
        for width in mlp:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim + prev, 1)

    def forward(self, x_idx: torch.Tensor) -> torch.Tensor:
        e = self.emb(x_idx).flatten(1)          # (B, n_fields * k)
        x0 = e
        xl = e
        for w, b in zip(self.cross_w, self.cross_b):
            xl = x0 * (xl @ w) + b + xl
        deep = self.mlp(e)
        return self.head(torch.cat([xl, deep], dim=1)).squeeze(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prediction-path", default=None)
    ap.add_argument("--trial-config", default=None)
    ap.add_argument("--submission-path", default=None)
    a = ap.parse_args()

    trial: dict = {}
    if a.trial_config:
        with open(a.trial_config, encoding="utf-8") as fh:
            trial = json.load(fh)
    allowed = {"k", "lr", "l2", "epochs", "batch_size", "patience"}
    unknown = sorted(set(trial) - allowed)
    if unknown:
        raise ValueError({"unknown_trial_config_keys": unknown})

    seed_everything(a.seed)

    splits = load(a.data_dir)
    if len(splits["train"]) != 1_141_112 or len(splits["valid"]) != 124_909:
        raise RuntimeError({"train": len(splits["train"]), "valid": len(splits["valid"])})

    enc, dim = encode(splits)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DCN(dim, Xtr.shape[1], k=int(trial.get("k", 16))).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(trial.get("lr", 1e-3)),
        weight_decay=float(trial.get("l2", 1e-6)),
    )
    loss_fn = nn.BCEWithLogitsLoss()

    Xtr_t = torch.from_numpy(Xtr.astype(np.int64))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    Xva_t = torch.from_numpy(Xva.astype(np.int64)).to(device)

    batch_size = int(trial.get("batch_size", 8192))
    patience = int(trial.get("patience", 4))
    generator = torch.Generator().manual_seed(a.seed)

    def predict(x_np: np.ndarray) -> np.ndarray:
        model.eval()
        out = []
        xt = torch.from_numpy(x_np.astype(np.int64))
        with torch.no_grad():
            for i in range(0, len(xt), 200_000):
                chunk = xt[i:i + 200_000].to(device)
                out.append(torch.sigmoid(model(chunk)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    best_primary, best_state, bad = -1.0, None, 0
    for _ in range(int(trial.get("epochs", 15))):
        model.train()
        order = torch.randperm(len(ytr_t), generator=generator)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        primary = evaluate(uva, yva, predict(Xva))["primary"]
        if primary > best_primary + 1e-5:
            best_primary, bad = primary, 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if a.prediction_path:
        scores = predict(Xva)
        write_validation_predictions(a.prediction_path, splits["valid"], scores)
        print(json.dumps({"status": "predictions_written", "rows": len(scores)}))
    if a.submission_path:
        write_hidden_submission(a.submission_path, splits, enc, predict)


if __name__ == "__main__":
    main()
