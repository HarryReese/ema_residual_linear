import argparse
import json
from pathlib import Path

from engine import evaluate, train


ROOT = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--deterministic", type=int, default=0)
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--experiment_name", default="core/case6_seq750_pred100")
    parser.add_argument(
        "--checkpoint_path",
        default="./results/core/case6_seq750_pred100/seed_2026/best_model.pth",
    )
    parser.add_argument("--root_path", default="./dataset")
    parser.add_argument("--data_path", default="case6.csv")
    parser.add_argument(
        "--target_cols",
        default="heave(m),roll(deg),pitch(deg)",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=750)
    parser.add_argument("--pred_len", type=int, default=100)
    parser.add_argument("--enc_in", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--d_ff", type=int, default=192)
    parser.add_argument("--e_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.001)
    parser.add_argument("--seasonal_patch_len", type=int, default=64)
    parser.add_argument("--seasonal_stride", type=int, default=32)
    parser.add_argument("--ema_alpha", type=float, default=0.3)
    parser.add_argument("--mamba_d_state", type=int, default=16)
    parser.add_argument("--mamba_d_conv", type=int, default=4)
    parser.add_argument("--mamba_expand", type=int, default=2)
    parser.add_argument("--pm_channel_factor", type=int, default=2)
    parser.add_argument("--pm_dropout", type=float, default=0.1)
    parser.add_argument("--channel_beta_init", type=float, default=1.0)
    parser.add_argument("--train_epochs", type=int, default=150)
    parser.add_argument("--learning_rate", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--early_stop_delta", type=float, default=1e-5)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--min_lr", type=float, default=2e-6)
    parser.add_argument("--lr_threshold", type=float, default=1e-4)
    parser.add_argument("--arctan_m", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def result_directory(args: argparse.Namespace) -> Path:
    base = Path(args.results_dir)
    if not base.is_absolute():
        base = ROOT / base
    return base / args.experiment_name / f"seed_{args.seed}"


def main() -> None:
    args = parse_args()
    result_dir = result_directory(args)
    result_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(vars(args), sort_keys=True, ensure_ascii=False))
    if args.mode == "train":
        train(args, result_dir)
    else:
        evaluate(args, result_dir)


if __name__ == "__main__":
    main()
