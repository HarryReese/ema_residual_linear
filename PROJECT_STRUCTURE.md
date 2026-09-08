# Project Structure

```text
ema_residual_linear/
├── model.py                 final EMA-residual Mamba + Linear model
├── layers/
│   ├── __init__.py
│   └── model_layers.py      RevIN, EMA, patching, Channel-MLP and Mamba
├── data.py                  chronological split and train-only scaling
├── engine.py                training, validation and evaluation
├── run.py                   command-line entry point
├── scripts/
│   ├── train.sh
│   └── evaluate.sh
├── requirements.txt
├── THIRD_PARTY_NOTICES.md
└── README.md
```
