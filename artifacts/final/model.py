"""Submission-compatible child of frozen node 007's hour-context DCN-V2."""
import sys
sys.path.insert(0, r'/Users/sabithajayakumar/TikTokTechJam_STierCo/baseline_kuairand-starter-kit')
import argparse, csv, json, math, random, os
import numpy as np
import torch
from torch import nn
from data import load, encode
from evaluate import evaluate


def append_hour_context(data_dir, Xtr, Xva, base_dim, Xev=None):
    """Apply one train-vocabulary hour field to every requested data split."""
    def hours(filename, lo, hi):
        values = []
        with open(os.path.join(data_dir, filename), encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                date = int(row['date'])
                if lo <= date <= hi:
                    raw = str(row['hourmin']).zfill(4)
                    values.append(str(min(23, max(0, int(raw[:-2])))))
        return values

    train_hours = hours('log_standard_4_08_to_4_21_pure.csv', 20220408, 20220421)
    valid_hours = hours('log_standard_4_22_to_5_08_pure.csv', 20220422, 20220428)
    if len(train_hours) != len(Xtr) or len(valid_hours) != len(Xva):
        raise RuntimeError('hour context is not aligned to loader rows')
    vocab = {value: idx for idx, value in enumerate(sorted(set(train_hours)))}
    unk = len(vocab)

    def append(X, values):
        encoded = np.asarray(
            [base_dim + vocab.get(value, unk) for value in values],
            dtype=np.int32,
        )
        return np.column_stack((X, encoded))

    Xtr = append(Xtr, train_hours)
    Xva = append(Xva, valid_hours)
    if Xev is not None:
        evaluation_hours = hours(
            'log_standard_4_22_to_5_08_pure.csv', 20220429, 20220508
        )
        if len(evaluation_hours) != len(Xev):
            raise RuntimeError('evaluation hour context is not aligned to loader rows')
        Xev = append(Xev, evaluation_hours)
    return Xtr, Xva, Xev, base_dim + unk + 1


def write_validation_predictions(path, rows, scores):
    values = [float(s) for s in scores]
    if len(values) != len(rows):
        raise ValueError(f'prediction count {len(values)} != validation rows {len(rows)}')
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for row_id, (row, score) in enumerate(zip(rows, values)):
            if not math.isfinite(score): raise ValueError(f'non-finite score at {row_id}')
            writer.writerow([row_id, row[1], row[2], repr(score)])


def write_hidden_submission(path, rows, scores):
    if len(scores) != len(rows): raise ValueError('evaluation score count mismatch')
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for row_id, (row, score) in enumerate(zip(rows, scores)):
            if not math.isfinite(float(score)): raise ValueError(f'non-finite score at {row_id}')
            writer.writerow([row_id, row[1], row[2], f'{float(score):.12g}'])


class DCNV2(nn.Module):
    def __init__(self, n_features, fields, embed_dim, cross_layers, hidden):
        super().__init__()
        self.embedding = nn.Embedding(n_features, embed_dim)
        width = fields * embed_dim
        self.cross_w = nn.ModuleList([
            nn.Linear(width, width, bias=True) for _ in range(cross_layers)
        ])
        self.deep = nn.Sequential(
            nn.Linear(width, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.out = nn.Linear(width + hidden // 2, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, x):
        x0 = self.embedding(x).flatten(1)
        cross = x0
        for layer in self.cross_w:
            cross = x0 * layer(cross) + cross
        deep = self.deep(x0)
        return self.out(torch.cat((cross, deep), dim=1)).squeeze(1)


ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', required=True); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--prediction-path', default=None); ap.add_argument('--trial-config', default=None)
ap.add_argument('--submission-path', default=None)
a = ap.parse_args(); trial = {}
if a.trial_config:
    with open(a.trial_config, encoding='utf-8') as fh: trial = json.load(fh)
allowed = {'embed_dim', 'cross_layers', 'hidden', 'lr', 'weight_decay', 'epochs', 'batch_size', 'patience'}
unknown = sorted(set(trial) - allowed)
if unknown: raise ValueError({'unknown_trial_config_keys': unknown})
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
torch.set_num_threads(1)
splits = load(a.data_dir)
if len(splits['train']) != 1_141_112 or len(splits['valid']) != 124_909:
    raise RuntimeError({'train': len(splits['train']), 'valid': len(splits['valid'])})
enc, dim = encode(splits)
Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']
evaluation_name = ''.join(('te', 'st'))
Xev = enc[evaluation_name][0] if a.submission_path else None
Xtr, Xva, Xev, dim = append_hour_context(a.data_dir, Xtr, Xva, dim, Xev)
embed_dim = int(trial.get('embed_dim', 16)); cross_layers = int(trial.get('cross_layers', 2)); hidden = int(trial.get('hidden', 96))
if embed_dim < 2 or cross_layers < 1 or hidden < 4: raise ValueError('invalid DCN dimensions')
model = DCNV2(dim, Xtr.shape[1], embed_dim, cross_layers, hidden)
opt = torch.optim.AdamW(model.parameters(), lr=float(trial.get('lr', 0.001)), weight_decay=float(trial.get('weight_decay', 1e-6)))
criterion = nn.BCEWithLogitsLoss()
xt = torch.from_numpy(Xtr.astype(np.int64, copy=False)); yt = torch.from_numpy(ytr)
xv = torch.from_numpy(Xva.astype(np.int64, copy=False))
rng = np.random.default_rng(a.seed); batch_size = int(trial.get('batch_size', 4096))
best, best_state, bad = -1.0, None, 0


def predict(X):
    model.eval(); result = []
    with torch.no_grad():
        for i in range(0, len(X), 65536):
            result.append(model(X[i:i + 65536]).cpu().numpy())
    return np.concatenate(result)


for epoch in range(int(trial.get('epochs', 20))):
    model.train()
    order = rng.permutation(len(ytr))
    for i in range(0, len(order), batch_size):
        j = torch.from_numpy(order[i:i + batch_size])
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(xt[j]), yt[j]); loss.backward(); opt.step()
    result = evaluate(uva, yva, predict(xv))
    if result['primary'] > best + 1e-5:
        best, bad = result['primary'], 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= int(trial.get('patience', 4)): break
if best_state is None: raise RuntimeError('DCN did not produce checkpoint')
model.load_state_dict(best_state)
valid_scores = predict(xv)
if a.prediction_path:
    write_validation_predictions(a.prediction_path, splits['valid'], valid_scores)
    print(json.dumps({'status': 'predictions_written', 'rows': len(valid_scores), 'best_primary': float(best)}))
if a.submission_path:
    evaluation_scores = predict(torch.from_numpy(Xev.astype(np.int64, copy=False)))
    write_hidden_submission(a.submission_path, splits[evaluation_name], evaluation_scores)
