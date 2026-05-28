import argparse
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# User configuration — plug in any tabular classification dataset here
# ---------------------------------------------------------------------------
dataset = "higgs_44129_sample_50000.csv"
target: str = "target"
id_col: Optional[str] = None
max_cardinality = 20  # integer columns with <= this many unique values -> categorical


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    target_col: str,
    id_col_name: Optional[str],
) -> List[str]:
    excluded = {target_col}
    if id_col_name is not None:
        if id_col_name not in df.columns:
            raise ValueError(f"id_col '{id_col_name}' not found in dataset columns.")
        excluded.add(id_col_name)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

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
    target_col: str
    numerical_cols: List[str]
    categorical_cols: List[str]
    dummy_columns: List[str] = field(default_factory=list)
    x_mean: Optional[np.ndarray] = None
    x_std: Optional[np.ndarray] = None
    class_labels: Optional[List[str]] = None

    @property
    def num_features(self) -> int:
        return len(self.numerical_cols) + len(self.dummy_columns)

    @property
    def num_classes(self) -> int:
        if self.class_labels is None:
            raise RuntimeError("class_labels not initialized. Call fit_transform first.")
        return len(self.class_labels)

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
        raw = df[self.target_col].astype(str).values
        if fit:
            self.class_labels = sorted(set(raw))
        if self.class_labels is None:
            raise RuntimeError("class_labels not initialized. Call fit_transform first.")
        label_to_idx = {lab: idx for idx, lab in enumerate(self.class_labels)}
        encoded = np.array([label_to_idx.get(v, -1) for v in raw], dtype=np.int64)
        unknown = int(np.sum(encoded == -1))
        if unknown > 0:
            raise ValueError(
                f"{unknown} samples have target values not seen during fit. "
                f"Known classes: {self.class_labels}"
            )
        return encoded

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
    target_col: str,
    id_col_name: Optional[str],
    cardinality_threshold: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, TabularPreprocessor]:
    df = load_dataframe(file_path)
    feature_cols = infer_feature_columns(df, target_col, id_col_name)

    work_df = df[feature_cols + [target_col]].copy()
    work_df = work_df.dropna(axis=0, how="any")
    if work_df.empty:
        raise ValueError("No rows remain after dropping missing values.")

    numerical_cols, categorical_cols = infer_column_types(
        work_df, feature_cols, cardinality_threshold
    )
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Target column: {target_col}")

    train_df, val_df, test_df = split_dataframe(work_df, seed=seed)
    preprocessor = TabularPreprocessor(
        target_col=target_col,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
    )

    x_train, y_train = preprocessor.fit_transform(train_df)
    x_val, y_val = preprocessor.transform(val_df)
    x_test, y_test = preprocessor.transform(test_df)

    print(f"Encoded feature tokens: {preprocessor.num_features}")
    print(f"Distinct classes ({preprocessor.num_classes}): {preprocessor.class_labels}")
    train_dist = Counter(y_train.tolist())
    print(f"Train class distribution: {dict(sorted(train_dist.items()))}")
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
        torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long()
    )
    test_ds = TensorDataset(
        torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long()
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


class MLPHead(nn.Module):
    """
    Conventional MLP nonlinear predictor after Transformer pooling.
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderMLPClassifier(nn.Module):
    """
    Feature tokens -> Transformer Encoder (MLP FFN) -> mean pooling -> MLP head -> linear(num_classes)
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        device: torch.device,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        kan_hidden_dim: int = 64,
        dropout: float = 0.2,
        kan_grid: int = 3,
        kan_k: int = 3,
        kan_seed: int = 42,
        ff_chunk_size: int = 512,
        head_chunk_size: int = 512,
    ) -> None:
        super().__init__()
        del kan_grid, kan_k, kan_seed, ff_chunk_size, head_chunk_size
        self.num_features = num_features
        self.device_ref = device
        self.d_model = d_model

        self.feature_value_proj = nn.Linear(1, d_model)
        self.feature_id_embed = nn.Embedding(num_features, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=kan_hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.mlp_head = MLPHead(d_model=d_model, hidden_dim=kan_hidden_dim, dropout=dropout)
        self.cls_head = nn.Linear(kan_hidden_dim, num_classes)

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
        nonlinear = self.mlp_head(pooled)
        logits = self.cls_head(nonlinear)  # (B, num_classes)
        return logits


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
    criterion: nn.Module,
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
        logits = model(xb)
        loss = criterion(logits, yb)
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
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_preds: List[torch.Tensor] = []
    all_trues: List[torch.Tensor] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_count += bs
            all_preds.append(logits.argmax(dim=1).cpu())
            all_trues.append(yb.cpu())

    pred_np = torch.cat(all_preds, dim=0).numpy()
    true_np = torch.cat(all_trues, dim=0).numpy()
    return total_loss / max(total_count, 1), pred_np, true_np


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    accuracy = float(np.mean(y_true == y_pred))

    per_class_precision: List[float] = []
    per_class_recall: List[float] = []
    per_class_f1: List[float] = []

    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    macro_precision = float(np.mean(per_class_precision))
    macro_recall = float(np.mean(per_class_recall))
    macro_f1 = float(np.mean(per_class_f1))

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_confusion_matrix(
    cm: np.ndarray,
    class_labels: List[str],
) -> None:
    max_label_len = max(len(lab) for lab in class_labels)
    header = " " * (max_label_len + 2) + "  ".join(
        f"{lab:>{max_label_len}}" for lab in class_labels
    )
    print(header)
    for i, lab in enumerate(class_labels):
        row = "  ".join(f"{cm[i, j]:>{max_label_len}}" for j in range(len(class_labels)))
        print(f"{lab:>{max_label_len}}  {row}")


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
        description="Train/Evaluate dataset-agnostic Transformer Encoder - MLP tabular classifier."
    )
    parser.add_argument("--dataset", type=str, default=dataset)
    parser.add_argument(
        "--target",
        type=str,
        default=target,
        help="Target column name (single classification target).",
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
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="best_model_mlp_cl.pt",
        help="Path to save/load the best validation checkpoint.",
    )
    parser.add_argument(
        "--lr_plot_path",
        type=str,
        default="tra-enc-mlp-tabular-cl-lr.png",
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
        target_col=args.target,
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

    model = TransformerEncoderMLPClassifier(
        num_features=preprocessor.num_features,
        num_classes=preprocessor.num_classes,
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

    criterion = nn.CrossEntropyLoss()

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
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            global_step_start=global_step,
            log_every=max(1, args.log_every),
        )
        global_step = epoch_stats.global_step

        val_loss, val_preds, val_trues = evaluate(
            model=model, loader=val_loader, criterion=criterion, device=device
        )
        val_acc = float(np.mean(val_preds == val_trues))
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
            f"val_acc={val_acc:.4f} | "
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

    best_train_loss, train_preds, train_trues = evaluate(
        model=model, loader=train_loader, criterion=criterion, device=device
    )
    best_val_loss_eval, val_preds_best, val_trues_best = evaluate(
        model=model, loader=val_loader, criterion=criterion, device=device
    )
    final_test_loss, test_preds, test_trues = evaluate(
        model=model, loader=test_loader, criterion=criterion, device=device
    )

    train_metrics = classification_metrics(train_trues, train_preds, preprocessor.num_classes)
    val_metrics = classification_metrics(val_trues_best, val_preds_best, preprocessor.num_classes)
    test_metrics = classification_metrics(test_trues, test_preds, preprocessor.num_classes)
    cm = confusion_matrix(test_trues, test_preds, preprocessor.num_classes)

    generalization_gap = best_train_loss - best_val_loss_eval
    flops_per_forward = 2 * total_params * preprocessor.num_features
    flops_per_batch = flops_per_forward * args.batch_size
    threshold_step_str = str(threshold_step) if threshold_step is not None else "Not reached"
    stopped_early_str = "Yes" if stopped_early else "No"
    plot_lr_history(lr_history, args.lr_plot_path)

    print("\n=================================================")
    print("MODEL: Transformer Encoder - MLP (Tabular Classification)")
    print("=================================================")
    print(f"Dataset: {args.dataset}")
    print(f"Target: {preprocessor.target_col}")
    print(f"Classes ({preprocessor.num_classes}): {preprocessor.class_labels}")
    print(f"Feature tokens: {preprocessor.num_features}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}\n")

    print(f"Best Training Loss (CE): {best_train_loss:.6f}")
    print(f"Best Validation Loss (CE): {best_val_loss_eval:.6f}")
    print(f"Test Loss (CE, best checkpoint): {final_test_loss:.6f}\n")

    print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Val   Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Test  Accuracy: {test_metrics['accuracy']:.4f}\n")

    print(f"Test Macro Precision: {test_metrics['macro_precision']:.4f}")
    print(f"Test Macro Recall:    {test_metrics['macro_recall']:.4f}")
    print(f"Test Macro F1:        {test_metrics['macro_f1']:.4f}\n")

    print("Test Confusion Matrix (rows=true, cols=predicted):")
    print_confusion_matrix(cm, preprocessor.class_labels)
    print()

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
