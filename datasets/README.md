# Datasets

This directory documents the datasets used by the project.

| Dataset | Role | Status |
|---|---|---|
| KuaiRand-Pure | Required benchmark | Supported |
| KuaiRand-1k | Bonus benchmark | Download instructions to be added when supported |
| KuaiRand-27k | Bonus benchmark | Download instructions to be added when supported |

Only KuaiRand datasets may be used as training data for this challenge. Do not add external training data.

## KuaiRand-Pure

Download the official archive from Zenodo. Run these commands from the repository root.

### macOS and Linux

```bash
curl -L https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz \
  -o datasets/KuaiRand-Pure.tar.gz
tar -xzf datasets/KuaiRand-Pure.tar.gz -C datasets
```

### Windows PowerShell

```powershell
Invoke-WebRequest `
  -Uri "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz" `
  -OutFile "datasets\KuaiRand-Pure.tar.gz"
tar -xzf datasets\KuaiRand-Pure.tar.gz -C datasets
```

After extraction, the required files should have this layout:

```text
datasets/
├── README.md
└── KuaiRand-Pure/
    ├── LICENSE
    ├── load_data_pure.py
    └── data/
        ├── log_random_4_22_to_5_08_pure.csv
        ├── log_standard_4_08_to_4_21_pure.csv
        ├── log_standard_4_22_to_5_08_pure.csv
        ├── user_features_pure.csv
        ├── video_features_basic_pure.csv
        └── video_features_statistic_pure.csv
```

To reproduce the organizer's Factorization Machine baseline from the repository root:

```bash
python baseline_kuairand-starter-kit/baseline.py \
  --model fm \
  --data_dir datasets/KuaiRand-Pure/data
```

The published validation primary score is approximately `0.6016`. See `baseline_kuairand-starter-kit/baseline_scores.json` for the complete reference scores and convergence settings.

## Bonus datasets

KuaiRand-1k and KuaiRand-27k are optional bonus benchmarks. Their exact download, extraction, and verification commands will be added here after support for each dataset is implemented and tested.

Use only official downloads linked from [kuairand.com](https://kuairand.com). Record the source URL, archive checksum, expected directory structure, and verification command when adding either dataset.

## Repository policy

- Do not commit downloaded archives or extracted dataset contents.
- Keep the dataset directory names shown in this document so scripts can use stable paths.
- Track code, configuration, download instructions, and small reproducibility metadata instead.
- Preserve the license included with each local dataset download.
