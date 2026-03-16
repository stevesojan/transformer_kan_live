import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load UCI Energy Efficiency data from ENB2012_data.xlsx.
    Uses first 8 numeric columns as features and next 2 numeric columns as targets.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_excel(path)
    df = df.dropna(axis=0, how="all")
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=0, how="any")

    if numeric_df.shape[1] < 10:
        raise ValueError(
            f"Expected at least 10 numeric columns, got {numeric_df.shape[1]}."
        )

    values = numeric_df.iloc[:, :10].to_numpy(dtype=np.float32)
    x = values[:, :8]
    y = values[:, 8:10]
    return x, y


def split_data(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic split:
    - Train: first 75%
    - Val: next 5%
    - Test: last 20%
    """
    n = x.shape[0]
    n_train = int(0.75 * n)
    n_val = int(0.05 * n)

    x_train = x[:n_train]
    y_train = y[:n_train]
    x_val = x[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]
    x_test = x[n_train + n_val :]
    y_test = y[n_train + n_val :]
    return x_train, y_train, x_val, y_val, x_test, y_test


def standardize(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    eps = 1e-8
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + eps
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + eps

    x_train_n = (x_train - x_mean) / x_std
    x_val_n = (x_val - x_mean) / x_std
    x_test_n = (x_test - x_mean) / x_std

    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    return (
        x_train_n,
        y_train_n,
        x_val_n,
        y_val_n,
        x_test_n,
        y_test_n,
        y_mean,
        y_std,
    )


def make_dataloaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TensorDataset(
        torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
    )
    val_ds = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float())
    test_ds = TensorDataset(
        torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader, test_loader


class KANFeedForward(nn.Module):
    """
    KAN replacement for Transformer encoder MLP blocks.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        kan_grid: int,
        kan_k: int,
        dropout: float,
        kan_seed: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.chunk_size = chunk_size

        try:
            from kan import KAN  # type: ignore
        except ImportError:
            import sys

            script_dir = Path(__file__).resolve().parent
            pykan_root = script_dir / "pykan"
            if pykan_root.exists():
                sys.path.append(str(pykan_root))
                from kan import KAN  # type: ignore
            else:
                raise ImportError(
                    "Could not import KAN. Install pykan or place local repo at ./pykan."
                )

        self.kan = KAN(
            width=[d_model, hidden_dim, d_model],
            grid=kan_grid,
            k=kan_k,
            base_fun="silu",
            symbolic_enabled=False,
            save_act=False,
            auto_save=False,
            seed=kan_seed,
        )
        self.kan.speed()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        bsz, seq_len, d_model = x.shape
        x = x.contiguous()
        flat = x.reshape(-1, d_model)

        if self.chunk_size <= 0 or flat.size(0) <= self.chunk_size:
            out = self.kan(flat)
        else:
            out = torch.empty_like(flat)
            for i in range(0, flat.size(0), self.chunk_size):
                j = i + self.chunk_size
                out[i:j] = self.kan(flat[i:j])

        out = out.reshape(bsz, seq_len, d_model)
        out = self.dropout(out)
        return out


class KANHead(nn.Module):
    """
    KAN nonlinear predictor after Transformer pooling.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        kan_grid: int,
        kan_k: int,
        dropout: float,
        kan_seed: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.chunk_size = chunk_size

        try:
            from kan import KAN  # type: ignore
        except ImportError:
            import sys

            script_dir = Path(__file__).resolve().parent
            pykan_root = script_dir / "pykan"
            if pykan_root.exists():
                sys.path.append(str(pykan_root))
                from kan import KAN  # type: ignore
            else:
                raise ImportError(
                    "Could not import KAN. Install pykan or place local repo at ./pykan."
                )

        self.kan = KAN(
            width=[d_model, hidden_dim, hidden_dim],
            grid=kan_grid,
            k=kan_k,
            base_fun="silu",
            symbolic_enabled=False,
            save_act=False,
            auto_save=False,
            seed=kan_seed,
        )
        self.kan.speed()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C)
        x = x.contiguous()
        if self.chunk_size <= 0 or x.size(0) <= self.chunk_size:
            out = self.kan(x)
        else:
            out = torch.empty((x.size(0), self.kan.width_out[-1]), device=x.device, dtype=x.dtype)
            for i in range(0, x.size(0), self.chunk_size):
                j = i + self.chunk_size
                out[i:j] = self.kan(x[i:j])
        out = self.dropout(out)
        return out


class TransformerEncoderLayerKAN(nn.Module):
    """
    Transformer encoder layer with KAN replacing FFN.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        kan_hidden_dim: int,
        dropout: float,
        kan_grid: int,
        kan_k: int,
        kan_seed: int,
        ff_chunk_size: int,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.kan_ff = KANFeedForward(
            d_model=d_model,
            hidden_dim=kan_hidden_dim,
            kan_grid=kan_grid,
            kan_k=kan_k,
            dropout=dropout,
            kan_seed=kan_seed,
            chunk_size=ff_chunk_size,
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = src
        attn_out = self.self_attn(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.kan_ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class TransformerEncoderKANRegressor(nn.Module):
    """
    Feature tokens -> Transformer Encoder (KAN FFN) -> mean pooling -> KAN head -> linear(2)
    """

    def __init__(
        self,
        num_features: int,
        output_dim: int,
        device: torch.device,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        kan_hidden_dim: int = 128,
        dropout: float = 0.1,
        kan_grid: int = 3,
        kan_k: int = 3,
        kan_seed: int = 42,
        ff_chunk_size: int = 512,
        head_chunk_size: int = 512,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.device_ref = device
        self.d_model = d_model

        # Scalar feature -> token embedding
        self.feature_value_proj = nn.Linear(1, d_model)
        self.feature_id_embed = nn.Embedding(num_features, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = TransformerEncoderLayerKAN(
            d_model=d_model,
            nhead=nhead,
            kan_hidden_dim=kan_hidden_dim,
            dropout=dropout,
            kan_grid=kan_grid,
            kan_k=kan_k,
            kan_seed=kan_seed,
            ff_chunk_size=ff_chunk_size,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.kan_head = KANHead(
            d_model=d_model,
            hidden_dim=kan_hidden_dim,
            kan_grid=kan_grid,
            kan_k=kan_k,
            dropout=dropout,
            kan_seed=kan_seed,
            chunk_size=head_chunk_size,
        )
        self.reg_head = nn.Linear(kan_hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 8)
        bsz, feat_dim = x.shape
        if feat_dim != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {feat_dim}."
            )

        feature_ids = torch.arange(feat_dim, device=x.device).unsqueeze(0).expand(bsz, -1)
        tokens = self.feature_value_proj(x.unsqueeze(-1)) + self.feature_id_embed(feature_ids)
        tokens = self.embedding_dropout(tokens)

        encoded = self.encoder(tokens)  # (B, 8, d_model)
        pooled = encoded.mean(dim=1)  # (B, d_model)
        nonlinear = self.kan_head(pooled)
        out = self.reg_head(nonlinear)  # (B, 2)
        return out


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@dataclass
class EpochStats:
    loss: float
    epoch_time_sec: float
    samples_per_sec: float
    global_step: int


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    global_step_start: int,
    log_every: int,
) -> EpochStats:
    model.train()
    start_time = time.perf_counter()
    step_start = time.perf_counter()

    running_loss = 0.0
    seen = 0
    global_step = global_step_start

    for step, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        bs = xb.size(0)
        running_loss += loss.item() * bs
        seen += bs
        global_step += 1

        if (step % log_every == 0) or (step == len(loader) - 1):
            elapsed = time.perf_counter() - step_start
            avg_step = elapsed / max(step + 1, 1)
            print(
                f"  step {step:04d}/{len(loader)} | "
                f"loss={loss.item():.6f} | "
                f"avg_step_time={avg_step:.3f}s",
                flush=True,
            )

    epoch_time = time.perf_counter() - start_time
    avg_loss = running_loss / max(seen, 1)
    samples_per_sec = seen / max(epoch_time, 1e-9)
    return EpochStats(
        loss=avg_loss,
        epoch_time_sec=epoch_time,
        samples_per_sec=samples_per_sec,
        global_step=global_step,
    )


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    preds: List[torch.Tensor] = []
    trues: List[torch.Tensor] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_count += bs
            preds.append(pred.cpu())
            trues.append(yb.cpu())

    pred_np = torch.cat(preds, dim=0).numpy()
    true_np = torch.cat(trues, dim=0).numpy()
    return total_loss / max(total_count, 1), pred_np, true_np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(mse))

    sse = np.sum((y_true - y_pred) ** 2, axis=0)
    y_mean = np.mean(y_true, axis=0, keepdims=True)
    sst = np.sum((y_true - y_mean) ** 2, axis=0) + 1e-12
    r2_each = 1.0 - (sse / sst)
    r2 = float(np.mean(r2_each))

    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/Evaluate Transformer Encoder - KAN regressor on UCI Energy Efficiency."
    )
    parser.add_argument("--data_path", type=str, default="ENB2012_data.xlsx")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss_threshold", type=float, default=0.05)
    parser.add_argument("--log_every", type=int, default=10)

    # Match encoder configs from transformer_seqtoseq_kan.py
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--kan_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kan_grid", type=int, default=3)
    parser.add_argument("--kan_k", type=int, default=3)
    parser.add_argument("--kan_seed", type=int, default=42)
    parser.add_argument("--ff_chunk_size", type=int, default=512)
    parser.add_argument("--head_chunk_size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cpu_count = os.cpu_count()
    if cpu_count is not None and cpu_count > 0:
        torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(1)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print("Loading UCI Energy Efficiency dataset...")
    x, y = load_data(args.data_path)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x, y)
    (
        x_train_n,
        y_train_n,
        x_val_n,
        y_val_n,
        x_test_n,
        y_test_n,
        y_mean,
        y_std,
    ) = standardize(x_train, y_train, x_val, y_val, x_test, y_test)

    print(
        f"Split sizes | train={len(x_train_n)} | val={len(x_val_n)} | test={len(x_test_n)}"
    )
    train_loader, val_loader, test_loader = make_dataloaders(
        x_train_n,
        y_train_n,
        x_val_n,
        y_val_n,
        x_test_n,
        y_test_n,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = TransformerEncoderKANRegressor(
        num_features=8,
        output_dim=2,
        device=device,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        kan_hidden_dim=args.kan_hidden_dim,
        dropout=args.dropout,
        kan_grid=args.kan_grid,
        kan_k=args.kan_k,
        kan_seed=args.kan_seed,
        ff_chunk_size=args.ff_chunk_size,
        head_chunk_size=args.head_chunk_size,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_loss = float("inf")
    best_val_epoch = -1
    threshold_step: Optional[int] = None
    global_step = 0

    train_losses: List[float] = []
    val_losses: List[float] = []
    epoch_times: List[float] = []
    sample_rates: List[float] = []

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        epoch_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            global_step_start=global_step,
            log_every=max(1, args.log_every),
        )
        global_step = epoch_stats.global_step

        val_loss, _, _ = evaluate(model=model, loader=val_loader, device=device)

        train_losses.append(epoch_stats.loss)
        val_losses.append(val_loss)
        epoch_times.append(epoch_stats.epoch_time_sec)
        sample_rates.append(epoch_stats.samples_per_sec)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
        if threshold_step is None and val_loss < args.loss_threshold:
            threshold_step = global_step

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={epoch_stats.loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"time={epoch_stats.epoch_time_sec:.2f}s | "
            f"samples/s={epoch_stats.samples_per_sec:.2f}"
        )

    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    final_test_loss, y_pred_test_n, y_true_test_n = evaluate(
        model=model, loader=test_loader, device=device
    )

    y_pred_test = (y_pred_test_n * y_std) + y_mean
    y_true_test = (y_true_test_n * y_std) + y_mean
    test_metrics = regression_metrics(y_true_test, y_pred_test)

    generalization_gap = final_train_loss - final_val_loss
    flops_per_forward = 2 * total_params * 8
    flops_per_batch = flops_per_forward * args.batch_size
    threshold_step_str = str(threshold_step) if threshold_step is not None else "Not reached"

    print("\n=================================================")
    print("MODEL: Transformer Encoder - KAN (Tabular Regression)")
    print("=================================================")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}\n")
    print(f"Final Training Loss (MSE, normalized): {final_train_loss:.6f}")
    print(f"Final Validation Loss (MSE, normalized): {final_val_loss:.6f}")
    print(f"Final Test Loss (MSE, normalized): {final_test_loss:.6f}\n")
    print(f"Test MSE (original scale): {test_metrics['mse']:.6f}")
    print(f"Test MAE (original scale): {test_metrics['mae']:.6f}")
    print(f"Test RMSE (original scale): {test_metrics['rmse']:.6f}")
    print(f"Test R2 (mean over targets): {test_metrics['r2']:.6f}\n")
    print(f"Generalization Gap: {generalization_gap:.6f}\n")
    print(f"Best Validation Epoch: {best_val_epoch}")
    print(f"Loss Threshold Step: {threshold_step_str}\n")
    print(f"Training Time Per Epoch: {float(np.mean(epoch_times)):.4f} sec")
    print(f"Samples per Second: {float(np.mean(sample_rates)):.4f}")
    print(f"Estimated FLOPs per Batch: {flops_per_batch:.4e}")
    print("=================================================")


if __name__ == "__main__":
    main()
