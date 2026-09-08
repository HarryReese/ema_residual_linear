import json
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import (
    make_loaders,
    make_test_loader,
    scaler_from_state,
    scaler_state,
    target_columns,
)
from model import BestShipMotionModel


ROOT = Path(__file__).parent


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def horizon_weights(args: Namespace, device: torch.device) -> torch.Tensor:
    steps = torch.arange(
        1,
        args.pred_len + 1,
        device=device,
        dtype=torch.float32,
    )
    return -torch.atan(args.arctan_m * steps) + torch.pi / 4 + 1.0


def objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return torch.mean(torch.abs((prediction - target) * weights.view(1, -1, 1)))


def validation_loss(
    model: BestShipMotionModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    weights: torch.Tensor,
) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for inputs, targets in loader:
            prediction = model(inputs.float().to(device))
            losses.append(objective(prediction, targets.float().to(device), weights))
    return float(torch.stack(losses).mean().item())


def save_checkpoint(
    path: Path,
    model: BestShipMotionModel,
    scaler: dict[str, np.ndarray],
    epoch: int,
    val_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "scaler_state": scaler,
        },
        path,
    )


def train(args: Namespace, result_dir: Path) -> dict[str, object]:
    set_seed(args.seed, bool(args.deterministic))
    device = torch.device(f"cuda:{args.gpu}")
    weights = horizon_weights(args, device)
    train_dataset, loaders = make_loaders(args)
    model = BestShipMotionModel(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.lr_threshold,
        threshold_mode="rel",
        min_lr=args.min_lr,
    )
    checkpoint_path = result_dir / "best_model.pth"
    saved_scaler = scaler_state(train_dataset.scaler)
    best_loss = float("inf")
    best_epoch = 0
    no_improvement = 0

    for epoch in range(1, args.train_epochs + 1):
        model.train()
        losses = []
        for inputs, targets in loaders["train"]:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(inputs), targets, weights)
            loss.backward()
            if args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.detach())

        train_loss = float(torch.stack(losses).mean().item())
        val_loss = validation_loss(model, loaders["val"], device, weights)
        scheduler.step(val_loss)
        is_best = val_loss <= best_loss - args.early_stop_delta
        if is_best:
            best_loss = val_loss
            best_epoch = epoch
            no_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                saved_scaler,
                epoch,
                val_loss,
            )
        else:
            no_improvement += 1
        print(
            f"epoch={epoch:03d} train={train_loss:.9f} "
            f"val={val_loss:.9f} lr={optimizer.param_groups[0]['lr']:.8g} "
            f"is_best={is_best}"
        )
        if no_improvement >= args.patience:
            break

    print(f"best_epoch={best_epoch} best_val={best_loss:.9f}")
    return evaluate(args, result_dir, checkpoint_path)


def regression_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    true = true.astype(np.float64).reshape(-1)
    pred = pred.astype(np.float64).reshape(-1)
    error = pred - true
    mse = float(np.mean(error**2))
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(mse)),
        "r2": 1.0
        - float(np.sum(error**2))
        / float(np.sum((true - np.mean(true)) ** 2)),
    }


def evaluate(
    args: Namespace,
    result_dir: Path,
    checkpoint_path: Path | None = None,
) -> dict[str, object]:
    set_seed(args.seed, bool(args.deterministic))
    device = torch.device(f"cuda:{args.gpu}")
    path = checkpoint_path or Path(args.checkpoint_path)
    if not path.is_absolute():
        path = ROOT / path
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    scaler = scaler_from_state(checkpoint["scaler_state"])
    loader = make_test_loader(args, scaler)
    model = BestShipMotionModel(args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    predictions = []
    targets = []
    with torch.inference_mode():
        for inputs, batch_targets in loader:
            predictions.append(model(inputs.float().to(device)).cpu().numpy())
            targets.append(batch_targets.numpy())
    prediction_scaled = np.concatenate(predictions)
    target_scaled = np.concatenate(targets)
    prediction = scaler.inverse_transform(
        prediction_scaled.reshape(-1, args.enc_in)
    ).reshape(prediction_scaled.shape)
    target = scaler.inverse_transform(
        target_scaled.reshape(-1, args.enc_in)
    ).reshape(target_scaled.shape)
    channels = {
        name: regression_metrics(target[:, :, index], prediction[:, :, index])
        for index, name in enumerate(target_columns(args))
    }
    metrics = {
        "overall": regression_metrics(target, prediction),
        "channels": channels,
    }
    (result_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics
