import random
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def target_columns(args: Namespace) -> list[str]:
    return [item.strip() for item in args.target_cols.split(",") if item.strip()]


class ShipMotionDataset(Dataset):
    def __init__(
        self,
        split: str,
        args: Namespace,
        scaler: StandardScaler | None = None,
    ) -> None:
        frame = pd.read_csv(
            Path(args.root_path) / args.data_path,
            encoding="utf-8-sig",
        )
        values = frame[target_columns(args)].to_numpy(dtype=np.float32)
        train_end = int(len(values) * args.train_ratio)
        val_end = int(len(values) * (args.train_ratio + args.val_ratio))
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.scaler = scaler or StandardScaler().fit(values[:train_end])
        scaled = self.scaler.transform(values).astype(np.float32)
        start, end = {
            "train": (0, train_end),
            "val": (train_end - self.seq_len, val_end),
            "test": (val_end - self.seq_len, len(values)),
        }[split]
        self.values = scaled[start:end]

    def __len__(self) -> int:
        return len(self.values) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        middle = index + self.seq_len
        return (
            torch.from_numpy(self.values[index:middle]),
            torch.from_numpy(self.values[middle:middle + self.pred_len]),
        )


def seed_worker(_: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loaders(
    args: Namespace,
) -> tuple[ShipMotionDataset, dict[str, DataLoader]]:
    train = ShipMotionDataset("train", args)
    val = ShipMotionDataset("val", args, train.scaler)
    generator = torch.Generator().manual_seed(args.seed)
    return train, {
        "train": DataLoader(
            train,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
        ),
        "val": DataLoader(
            val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
        ),
    }


def scaler_from_state(state: Mapping[str, object]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(state["mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(state["scale"], dtype=np.float64)
    scaler.var_ = np.asarray(state["var"], dtype=np.float64)
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler


def scaler_state(scaler: StandardScaler) -> dict[str, np.ndarray]:
    return {
        "mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scale": np.asarray(scaler.scale_, dtype=np.float64),
        "var": np.asarray(scaler.var_, dtype=np.float64),
    }


def make_test_loader(args: Namespace, scaler: StandardScaler) -> DataLoader:
    return DataLoader(
        ShipMotionDataset("test", args, scaler),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
