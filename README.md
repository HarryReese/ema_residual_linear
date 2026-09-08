# EMA Residual Linear Ship Motion Prediction

This repository contains the pre-acceptance core implementation of the model
described in *A Lightweight Dual Branch Network with EMA Residual Learning for
Multistep Ship Motion Prediction*.

## Model

The model combines two complementary paths after reversible instance
normalization:

```text
Input [B, 750, 3]
  ├─ causal EMA residual -> Patch64/Stride32 -> Channel-MLP -> Mamba -> forecast
  └─ shared Linear(750, H) ------------------------------------------> forecast
                                      sum -> inverse RevIN -> output [B, H, 3]
```

The three channels are heave, roll, and pitch. The paper evaluates prediction
horizons of 20, 60, 100, 200, and 300 samples at a sampling interval of 0.05 s.

## Paper configuration

- history length: 750 samples (37.5 s)
- prediction horizons: 20, 60, 100, 200, and 300 samples
- hidden dimensions: `d_model=96`, `d_ff=192`
- residual patches: length 64, stride 32
- EMA coefficient: `alpha=0.3`
- temporal encoder: one Mamba layer with `d_state=16`, `d_conv=4`, `expand=2`
- channel interaction: learnable Channel-MLP coupling
- loss: arctan-weighted MAE
- chronological split: 70% train, 10% validation, 20% test
- batch size: 48; maximum epochs: 150

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

`CHECKPOINT_PATH`, `DATA_FILE`, `PRED_LEN`, and `SEED` can be overridden through
environment variables. Run `python run.py --help` for all core options.

## Release scope

This pre-acceptance release includes only the final model, the chronological
data pipeline, training, evaluation, checkpoint saving, test metrics, and two
portable entry scripts. Datasets, trained weights, detailed experiment records,
paper figures, ablation code, alternative backbones, intermediate-output
exports, complexity analysis, and latency analysis are reserved for the
complete reproducibility release.
