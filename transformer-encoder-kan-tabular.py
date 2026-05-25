import argparse
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# User configuration — plug in any tabular regression dataset here
# ---------------------------------------------------------------------------
dataset = "housing.csv"
target: Union[str, List[str]] = ["median_house_value"]
id_col: Optional[str] = None
max_cardinality = 20  # integer columns with <= this many unique values -> categorical


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_target_cols(target_cols: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(target_cols, str):
        return [target_cols]
    return list(target_cols)


def load_dataframe(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported dataset format '{suffix}'. Use csv, xlsx, xls, or parquet."
        )

    df = df.dropna(axis=0, how="all")
    return df


def infer_feature_columns(
    df: pd.DataFrame,
    target_cols: List[str],
    id_col_name: Optional[str],
) -> List[str]:
    excluded = set(target_cols)
    if id_col_name is not None:
        if id_col_name not in df.columns:
            raise ValueError(f"id_col '{id_col_name}' not found in dataset columns.")
        excluded.add(id_col_name)

    missing_targets = [col for col in target_cols if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Target column(s) not found in dataset: {missing_targets}")

    feature_cols = [col for col in df.columns if col not in excluded]
    if not feature_cols:
        raise ValueError("No feature columns remain after excluding target and id_col.")
    return feature_cols


def infer_column_types(
    df: pd.DataFrame,
    feature_cols: List[str],
    cardinality_threshold: int,
) -> Tuple[List[str], List[str]]:
    numerical_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in feature_cols:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            categorical_cols.append(col)
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            categorical_cols.append(col)
        elif pd.api.types.is_integer_dtype(series):
            n_unique = series.nunique(dropna=True)
            if n_unique <= cardinality_threshold:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        elif pd.api.types.is_float_dtype(series):
            numerical_cols.append(col)
        else:
            raise ValueError(f"Unsupported dtype for feature column '{col}': {series.dtype}")

    return numerical_cols, categorical_cols


@dataclass
class TabularPreprocessor:
    target_cols: List[str]
    numerical_cols: List[str]
    categorical_cols: List[str]
    dummy_columns: List[str] = field(default_factory=list)
    x_mean: Optional[np.ndarray] = None
    x_std: Optional[np.ndarray] = None
    y_mean: Optional[np.ndarray] = None
    y_std: Optional[np.ndarray] = None

    @property
    def num_features(self) -> int:
        return len(self.numerical_cols) + len(self.dummy_columns)

    @property
    def output_dim(self) -> int:
        return len(self.target_cols)

    def _encode_categorical_block(self, df: pd.DataFrame) -> np.ndarray:
        if not self.categorical_cols:
            return np.empty((len(df), 0), dtype=np.float32)

        encoded = pd.get_dummies(
            df[self.categorical_cols].astype(str),
            columns=self.categorical_cols,
            drop_first=False,
            dtype=float,
        )
        encoded = encoded.reindex(columns=self.dummy_columns, fill_value=0.0)
        return encoded.to_numpy(dtype=np.float32)

    def _encode_numerical_block(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        if not self.numerical_cols:
            return np.empty((len(df), 0), dtype=np.float32)

        values = df[self.numerical_cols].to_numpy(dtype=np.float32)
        if fit:
            eps = 1e-8
            self.x_mean = values.mean(axis=0, keepdims=True)
            self.x_std = values.std(axis=0, keepdims=True) + eps
        if self.x_mean is None or self.x_std is None:
            raise RuntimeError("Numerical standardization stats are not initialized.")
        return (values - self.x_mean) / self.x_std

    def _encode_targets(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        values = df[self.target_cols].to_numpy(dtype=np.float32)
        if fit:
            eps = 1e-8
            self.y_mean = values.mean(axis=0, keepdims=True)
            self.y_std = values.std(axis=0, keepdims=True) + eps
        if self.y_mean is None or self.y_std is None:
            raise RuntimeError("Target standardization stats are not initialized.")
        return (values - self.y_mean) / self.y_std

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.categorical_cols:
            train_dummies = pd.get_dummies(
                df[self.categorical_cols].astype(str),
                columns=self.categorical_cols,
                drop_first=False,
                dtype=float,
            )
            self.dummy_columns = train_dummies.columns.tolist()

        x_num = self._encode_numerical_block(df, fit=True)
        x_cat = self._encode_categorical_block(df)
        x = np.concatenate([x_num, x_cat], axis=1).astype(np.float32)
        y = self._encode_targets(df, fit=True)
        return x, y

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        x_num = self._encode_numerical_block(df, fit=False)
        x_cat = self._encode_categorical_block(df)
        x = np.concatenate([x_num, x_cat], axis=1).astype(np.float32)
        y = self._encode_targets(df, fit=False)
        return x, y

    def inverse_transform_targets(self, y_normalized: np.ndarray) -> np.ndarray:
        if self.y_mean is None or self.y_std is None:
            raise RuntimeError("Target standardization stats are not initialized.")
        return (y_normalized * self.y_std) + self.y_mean


def split_dataframe(
    df: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Shuffle, then split:
    - Train: first 75%
    - Val: next 5%
    - Test: last 20%
    """
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(0.75 * n)
    n_val = int(0.05 * n)

    train_df = shuffled.iloc[:n_train].copy()
    val_df = shuffled.iloc[n_train : n_train + n_val].copy()
    test_df = shuffled.iloc[n_train + n_val :].copy()
    return train_df, val_df, test_df


def prepare_tabular_data(
    file_path: str,
    target_cols: Union[str, Sequence[str]],
    id_col_name: Optional[str],
    cardinality_threshold: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, TabularPreprocessor]:
    df = load_dataframe(file_path)
    target_cols = normalize_target_cols(target_cols)
    feature_cols = infer_feature_columns(df, target_cols, id_col_name)

    work_df = df[feature_cols + target_cols].copy()
    work_df = work_df.dropna(axis=0, how="any")
    if work_df.empty:
        raise ValueError("No rows remain after dropping missing values.")

    numerical_cols, categorical_cols = infer_column_types(
        work_df, feature_cols, cardinality_threshold
    )
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Target columns ({len(target_cols)}): {target_cols}")

    train_df, val_df, test_df = split_dataframe(work_df, seed=seed)
    preprocessor = TabularPreprocessor(
        target_cols=target_cols,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
    )

    x_train, y_train = preprocessor.fit_transform(train_df)
    x_val, y_val = preprocessor.transform(val_df)
    x_test, y_test = preprocessor.transform(test_df)

    print(f"Encoded feature tokens: {preprocessor.num_features}")
    return x_train, y_train, x_val, y_val, x_test, y_test, preprocessor


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
    Feature tokens -> Transformer Encoder (KAN FFN) -> mean pooling -> KAN head -> linear(output_dim)
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
        # x: (B, num_features)
        bsz, feat_dim = x.shape
        if feat_dim != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {feat_dim}."
            )

        feature_ids = torch.arange(feat_dim, device=x.device).unsqueeze(0).expand(bsz, -1)
        tokens = self.feature_value_proj(x.unsqueeze(-1)) + self.feature_id_embed(feature_ids)
        tokens = self.embedding_dropout(tokens)

        encoded = self.encoder(tokens)  # (B, num_features, d_model)
        pooled = encoded.mean(dim=1)  # (B, d_model)
        nonlinear = self.kan_head(pooled)
        out = self.reg_head(nonlinear)  # (B, output_dim)
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
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(mse))

    sse = np.sum((y_true - y_pred) ** 2, axis=0)
    y_mean = np.mean(y_true, axis=0, keepdims=True)
    sst = np.sum((y_true - y_mean) ** 2, axis=0) + 1e-12
    r2_each = 1.0 - (sse / sst)
    r2 = float(np.mean(r2_each))

    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def plot_lr_history(lr_history: List[float], output_path: str) -> None:
    epochs = np.arange(1, len(lr_history) + 1)
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, lr_history, marker="o", linewidth=1.5, markersize=4)
    plt.title("Learning Rate per Epoch (ReduceLROnPlateau)")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/Evaluate dataset-agnostic Transformer Encoder - KAN tabular regressor."
    )
    parser.add_argument("--dataset", type=str, default=dataset)
    parser.add_argument(
        "--target",
        nargs="+",
        default=normalize_target_cols(target),
        help="One or more target column names.",
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default=id_col,
        help="Optional identifier column to exclude from features.",
    )
    parser.add_argument(
        "--max_cardinality",
        type=int,
        default=max_cardinality,
        help="Integer columns with <= this many unique values are treated as categorical.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss_threshold", type=float, default=0.05)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    parser.add_argument("--lr_scheduler_patience", type=int, default=3)
    parser.add_argument("--lr_scheduler_min_lr", type=float, default=1e-6)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="best_model.pt",
        help="Path to save/load the best validation checkpoint.",
    )
    parser.add_argument(
        "--lr_plot_path",
        type=str,
        default="transformer-encoder-kan-tabular-lr.png",
    )

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--kan_hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
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
    print(f"Loading dataset: {args.dataset}")
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        preprocessor,
    ) = prepare_tabular_data(
        file_path=args.dataset,
        target_cols=args.target,
        id_col_name=args.id_col,
        cardinality_threshold=args.max_cardinality,
        seed=args.seed,
    )

    print(
        f"Split sizes | train={len(x_train)} | val={len(x_val)} | test={len(x_test)}"
    )
    train_loader, val_loader, test_loader = make_dataloaders(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = TransformerEncoderKANRegressor(
        num_features=preprocessor.num_features,
        output_dim=preprocessor.output_dim,
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        min_lr=args.lr_scheduler_min_lr,
    )

    best_val_loss = float("inf")
    best_val_epoch = -1
    threshold_step: Optional[int] = None
    global_step = 0
    epochs_without_improvement = 0
    stopped_early = False

    train_losses: List[float] = []
    val_losses: List[float] = []
    epoch_times: List[float] = []
    sample_rates: List[float] = []
    lr_history: List[float] = []

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
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(epoch_stats.loss)
        val_losses.append(val_loss)
        epoch_times.append(epoch_stats.epoch_time_sec)
        sample_rates.append(epoch_stats.samples_per_sec)
        lr_history.append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.checkpoint_path)
        else:
            epochs_without_improvement += 1

        if threshold_step is None and val_loss < args.loss_threshold:
            threshold_step = global_step

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={epoch_stats.loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"lr={current_lr:.7f} | "
            f"time={epoch_stats.epoch_time_sec:.2f}s | "
            f"samples/s={epoch_stats.samples_per_sec:.2f}"
        )

        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            stopped_early = True
            break

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise RuntimeError(
            f"No checkpoint found at {args.checkpoint_path}. Training may have failed."
        )
    print(f"Loading best checkpoint from epoch {best_val_epoch}: {args.checkpoint_path}")
    try:
        state_dict = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    best_train_loss, _, _ = evaluate(model=model, loader=train_loader, device=device)
    best_val_loss_eval, _, _ = evaluate(model=model, loader=val_loader, device=device)
    final_test_loss, y_pred_test_n, y_true_test_n = evaluate(
        model=model, loader=test_loader, device=device
    )

    y_pred_test = preprocessor.inverse_transform_targets(y_pred_test_n)
    y_true_test = preprocessor.inverse_transform_targets(y_true_test_n)
    test_metrics = regression_metrics(y_true_test, y_pred_test)

    generalization_gap = best_train_loss - best_val_loss_eval
    flops_per_forward = 2 * total_params * preprocessor.num_features
    flops_per_batch = flops_per_forward * args.batch_size
    threshold_step_str = str(threshold_step) if threshold_step is not None else "Not reached"
    stopped_early_str = "Yes" if stopped_early else "No"
    plot_lr_history(lr_history, args.lr_plot_path)

    print("\n=================================================")
    print("MODEL: Transformer Encoder - KAN (Tabular Regression)")
    print("=================================================")
    print(f"Dataset: {args.dataset}")
    print(f"Targets: {preprocessor.target_cols}")
    print(f"Feature tokens: {preprocessor.num_features}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}\n")
    print(f"Best Training Loss (MSE, normalized): {best_train_loss:.6f}")
    print(f"Best Validation Loss (MSE, normalized): {best_val_loss_eval:.6f}")
    print(f"Test Loss (MSE, normalized, best checkpoint): {final_test_loss:.6f}\n")
    print(f"Test MSE (original scale): {test_metrics['mse']:.6f}")
    print(f"Test MAE (original scale): {test_metrics['mae']:.6f}")
    print(f"Test RMSE (original scale): {test_metrics['rmse']:.6f}")
    print(f"Test R2 (mean over targets): {test_metrics['r2']:.6f}\n")
    print(f"Generalization Gap: {generalization_gap:.6f}\n")
    print(f"Best Validation Epoch: {best_val_epoch}")
    print(f"Early Stopping Triggered: {stopped_early_str}")
    print(f"Checkpoint Path: {args.checkpoint_path}")
    print(f"Loss Threshold Step: {threshold_step_str}\n")
    print(f"Training Time Per Epoch: {float(np.mean(epoch_times)):.4f} sec")
    print(f"Samples per Second: {float(np.mean(sample_rates)):.4f}")
    print(f"Estimated FLOPs per Batch: {flops_per_batch:.4e}")
    print(f"LR Plot Path: {args.lr_plot_path}")
    print("=================================================")


if __name__ == "__main__":
    main()
