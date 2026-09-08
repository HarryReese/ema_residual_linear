# EMA Residual Linear Ship Motion Prediction

This repository contains  the model described in *A Lightweight Dual Branch Network with EMA Residual Learning for
Multistep Ship Motion Prediction*.



## Installation

Python 3.10 and an NVIDIA CUDA environment are required.

```bash
python -m pip install -r requirements.txt
```

## Data format

Datasets and trained weights are intentionally not included in this
pre-acceptance core release. Place a CSV file in `dataset/` with these columns:

```text
heave(m),roll(deg),pitch(deg)
```

The loader splits the continuous series in chronological order. The
`StandardScaler` is fitted only on the training interval and then reused for
validation and testing.

## Train

The default command runs the paper's 750-to-100 configuration on
`dataset/case6.csv`:

```bash
bash scripts/train.sh
```

Override the dataset or horizon without changing source files:

```bash
DATA_FILE=case4.csv PRED_LEN=300 SEED=2024 bash scripts/train.sh
```

The best checkpoint and test metrics are written to `results/`, which is
excluded from version control.

## Evaluate

After training, evaluate the generated best checkpoint with:

```bash
bash scripts/evaluate.sh
```



