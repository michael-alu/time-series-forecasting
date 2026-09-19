"""Training loop shared by all three models, plus timing for the report."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


import time
import platform
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from config import CellType, ScalingType
from dataset import Dataset
from models import Forecaster, build_model


@dataclass
class Config:
    """All the knobs for one training run."""

    cell: CellType = "LSTM"
    sequence_length: int = 144
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 8
    scaling: ScalingType = "Standard"
    seed: int = 42

    def label(self) -> str:
        return (
            f"{self.cell}_seq{self.sequence_length}_h{self.hidden_size}_{self.scaling}"
        )


@dataclass
class TrainedModel:
    """What a finished training run gives back."""

    model: Forecaster
    config: Config
    train_seconds: float
    epochs_run: int
    best_epoch: int
    best_val_loss: float
    parameters: int


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def describe_hardware(device: torch.device) -> dict[str, str]:
    """Recorded alongside the timings, since they mean nothing without it."""

    info = {
        "device": str(device),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }

    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)

    return info


def wait_for(device: torch.device) -> None:
    """GPU work is queued, not instant. Timings are wrong without this."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def warm_up(device: torch.device) -> None:
    """First GPU call pays setup costs. Burn that before timing anything."""
    model = build_model("LSTM", hidden_size=8).to(device)

    model(torch.zeros(4, 8, 1, device=device))

    wait_for(device)


def make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    tensors = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)


def train_model(
    dataset: Dataset,
    config: Config,
    device: torch.device | None = None,
    verbose: bool = True,
) -> TrainedModel:
    """Train until the validation loss stops improving, then keep the best weights."""
    device = device or pick_device()

    torch.manual_seed(config.seed)

    np.random.seed(config.seed)

    train_loader = make_loader(
        dataset["X_train"], dataset["y_train"], config.batch_size, True
    )

    val_loader = make_loader(
        dataset["X_val"], dataset["y_val"], config.batch_size, False
    )

    model = build_model(
        config.cell, config.hidden_size, config.num_layers, config.dropout, config.seed
    ).to(device)

    loss_function = nn.MSELoss()

    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = float("inf")

    best_epoch = 0

    best_weights = {k: v.clone() for k, v in model.state_dict().items()}

    epochs_without_improvement = 0

    wait_for(device)

    start = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        model.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimiser.zero_grad()

            loss = loss_function(model(xb), yb)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )  # stops exploding gradients

            optimiser.step()

        val_loss = validation_loss(model, val_loader, loss_function, device)

        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch

            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and epoch % 5 == 0:
            print(f"  epoch {epoch:3d}  val loss {val_loss:.5f}")

        if epochs_without_improvement >= config.patience:
            break

    wait_for(device)

    train_seconds = time.perf_counter() - start

    # keep the best epoch, not the last one
    model.load_state_dict(best_weights)

    return TrainedModel(
        model=model,
        config=config,
        train_seconds=train_seconds,
        epochs_run=epoch,
        best_epoch=best_epoch,
        best_val_loss=best_loss,
        parameters=model.parameter_count(),
    )


def validation_loss(
    model: Forecaster,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.eval()

    total, count = 0.0, 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            total += loss_function(model(xb), yb).item() * len(yb)

            count += len(yb)

    return total / count


def predict(
    model: Forecaster, X: np.ndarray, device: torch.device, repeats: int = 3
) -> tuple[np.ndarray, float]:
    """Predict, and time it. Returns scaled predictions and the median seconds taken."""
    model.eval()

    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=256)

    timings: list[float] = []

    outputs = np.array([])

    with torch.no_grad():
        for _ in range(repeats):
            chunks = []

            wait_for(device)

            start = time.perf_counter()

            for (xb,) in loader:
                chunks.append(model(xb.to(device)).cpu().numpy())

            wait_for(device)

            timings.append(time.perf_counter() - start)

            outputs = np.concatenate(chunks)

    return outputs, float(np.median(timings))
