from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


def normalize_weights(weights: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= eps:
        return np.ones_like(weights) / len(weights)
    return weights / total


def cosine_affinity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = max(np.linalg.norm(a) * np.linalg.norm(b), eps)
    return float(np.dot(a, b) / denom)


def knee_scores(obj_batch: np.ndarray) -> np.ndarray:
    objs = np.asarray(obj_batch, dtype=np.float64)
    if len(objs) == 0:
        return np.array([])
    mins = objs.min(axis=0, keepdims=True)
    spans = np.maximum(objs.max(axis=0, keepdims=True) - mins, 1e-8)
    norm = (objs - mins) / spans
    scores = np.zeros(len(norm), dtype=np.float64)
    for d in range(norm.shape[1]):
        order = np.argsort(norm[:, d])
        seq = norm[order]
        if len(seq) >= 3:
            curvature = np.zeros(len(seq), dtype=np.float64)
            curvature[1:-1] = np.linalg.norm(seq[2:] - 2 * seq[1:-1] + seq[:-2], axis=1)
            scores[order] += curvature
    scores /= max(norm.shape[1], 1)
    if np.allclose(scores.max(), 0.0):
        return np.ones_like(scores)
    return scores / (scores.max() + 1e-8)


def build_trace_matrix(trace: dict[str, np.ndarray], window: int = 8) -> np.ndarray:
    context = np.asarray(trace.get("context", []), dtype=np.float32)
    if context.ndim == 2 and len(context) > 0:
        seq = context
    else:
        obs = np.asarray(trace.get("obs", []), dtype=np.float32)
        if obs.ndim != 2 or len(obs) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        seq = obs
    if len(seq) < window:
        pad = np.repeat(seq[:1], window - len(seq), axis=0)
        seq = np.concatenate([pad, seq], axis=0)
    return seq


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class RegimeAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 48,
        nhead: int = 4,
        num_layers: int = 1,
        window: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window = window
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.context_head = nn.Linear(hidden_dim, hidden_dim)
        self.pred_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.context_head(pooled), self.pred_head(pooled)

    def embed(self, trace_matrix: np.ndarray, device: torch.device | None = None) -> np.ndarray:
        device = device or next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        x = torch.as_tensor(trace_matrix[-self.window :], dtype=dtype, device=device).unsqueeze(0)
        with torch.no_grad():
            context, _ = self.forward(x)
        return context.squeeze(0).cpu().numpy()

    def encode_with_forecast(
        self,
        trace_matrix: np.ndarray,
        device: torch.device | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        device = device or next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        seq = np.asarray(trace_matrix, dtype=np.float32)
        if len(seq) < self.window:
            pad = np.repeat(seq[:1], self.window - len(seq), axis=0)
            seq = np.concatenate([pad, seq], axis=0)
        current_window = seq[-self.window :]
        x = torch.as_tensor(current_window, dtype=dtype, device=device).unsqueeze(0)
        with torch.no_grad():
            current_context, pred = self.forward(x)
        pred_np = pred.squeeze(0).cpu().numpy()
        if self.window > 1:
            forecast_window = np.concatenate([current_window[1:], pred_np[None, :]], axis=0)
        else:
            forecast_window = pred_np[None, :]
        forecast_x = torch.as_tensor(forecast_window, dtype=dtype, device=device).unsqueeze(0)
        with torch.no_grad():
            forecast_context, _ = self.forward(forecast_x)
        return (
            current_context.squeeze(0).cpu().numpy(),
            forecast_context.squeeze(0).cpu().numpy(),
            pred_np,
        )

    def fit(
        self,
        traces: Iterable[dict[str, np.ndarray]],
        steps: int = 10,
        lr: float = 1e-3,
        device: torch.device | None = None,
    ) -> float:
        traces = list(traces)
        if not traces:
            return 0.0
        device = device or next(self.parameters()).device
        self.to(device)
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        dtype = next(self.parameters()).dtype
        losses = []
        for _ in range(steps):
            batch_loss = 0.0
            for trace in traces:
                seq = torch.as_tensor(
                    build_trace_matrix(trace, self.window), dtype=dtype, device=device
                )
                x = seq[-self.window :].unsqueeze(0)
                y = seq[-1:].mean(dim=0, keepdim=True)
                context, pred = self.forward(x)
                loss = torch.mean((pred - y) ** 2) + 0.1 * torch.mean(context ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                batch_loss += float(loss.detach().cpu())
            losses.append(batch_loss / len(traces))
        self.eval()
        return float(np.mean(losses))


def adapt_weights(
    base_weights: np.ndarray,
    gap_vector: np.ndarray,
    drift_score: float,
    knee_score: float,
) -> np.ndarray:
    base = normalize_weights(base_weights)
    gap = normalize_weights(np.maximum(gap_vector, 0.0))
    gate = float(1.0 / (1.0 + np.exp(-3.0 * drift_score)))
    knee_gate = 0.5 + 0.5 * float(knee_score)
    alpha = np.clip(gate * knee_gate, 0.0, 0.9)
    mixed = (1.0 - alpha) * base + alpha * gap
    return normalize_weights(mixed)
