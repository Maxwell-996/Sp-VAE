"""Learnable, format-agnostic token MoE adapters.

This module deliberately avoids a fixed Fourier or wavelet decomposition. It
keeps token correspondence intact and lets a content/coordinate-conditioned
router discover which experts are useful for each local representation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _tokenize(x: torch.Tensor) -> tuple[torch.Tensor, tuple[str, tuple[int, ...]]]:
    if x.dim() == 3:
        return x, ("tokens", tuple(x.shape))
    if x.dim() == 4:
        b, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(b, h * w, c), ("image", (h, w))
    if x.dim() == 5:
        b, c, t, h, w = x.shape
        return (
            x.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c),
            ("video", (t, h, w)),
        )
    raise ValueError(f"expected 3D/4D/5D input, got {tuple(x.shape)}")


def _restore(x: torch.Tensor, meta: tuple[str, tuple[int, ...]]) -> torch.Tensor:
    kind, shape = meta
    if kind == "tokens":
        return x
    b, _n, c = x.shape
    if kind == "image":
        h, w = shape
        return x.reshape(b, h, w, c).permute(0, 3, 1, 2)
    t, h, w = shape
    return x.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)


def _default_coords(
    meta: tuple[str, tuple[int, ...]],
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    kind, shape = meta
    if kind == "image":
        h, w = shape
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=device, dtype=dtype),
            torch.linspace(0, 1, w, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((torch.zeros_like(yy), yy, xx), dim=-1).reshape(n, 3)
    if kind == "video":
        t, h, w = shape
        tt, yy, xx = torch.meshgrid(
            torch.linspace(0, 1, t, device=device, dtype=dtype),
            torch.linspace(0, 1, h, device=device, dtype=dtype),
            torch.linspace(0, 1, w, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((tt, yy, xx), dim=-1).reshape(n, 3)
    index = torch.linspace(0, 1, n, device=device, dtype=dtype)
    return torch.stack((index, torch.zeros_like(index), torch.zeros_like(index)), dim=-1)


class _Expert(nn.Module):
    def __init__(self, hidden: int, expansion: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * expansion),
            nn.GELU(),
            nn.Linear(hidden * expansion, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnableTokenMoE(nn.Module):
    """Shared-plus-routed token MoE with optional sparse top-k gates.

    Inputs can be ``(B,N,C)``, ``(B,C,H,W)``, or ``(B,C,T,H,W)``. For point
    clouds or irregular meshes, callers can provide ``coords=(B,N,3)`` and a
    boolean ``mask=(B,N)`` without resampling to a regular grid. The residual
    projection is zero initialized so a newly attached adapter is initially an
    identity map.
    """

    def __init__(
        self,
        channels: int,
        *,
        hidden: int = 64,
        num_experts: int = 4,
        top_k: int = 2,
        expansion: int = 2,
        global_context: bool = False,
    ):
        super().__init__()
        if channels < 1 or hidden < 1:
            raise ValueError("channels and hidden must be positive")
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.global_context_enabled = bool(global_context)
        self.input = nn.Sequential(
            nn.LayerNorm(channels + 3),
            nn.Linear(channels + 3, hidden),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.router = nn.Linear(hidden * 2, num_experts)
        self.experts = nn.ModuleList(
            [_Expert(hidden, expansion=expansion) for _ in range(num_experts)]
        )
        self.shared = _Expert(hidden, expansion=expansion)
        self.global_expert = _Expert(hidden, expansion=expansion) if global_context else None
        self.output = nn.Linear(hidden, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self._balance_loss = torch.zeros(())
        self.register_buffer(
            "last_importance", torch.full((num_experts,), 1.0 / num_experts)
        )
        self.register_buffer("last_load", torch.full((num_experts,), 1.0 / num_experts))

    def _aux(
        self,
        coords: torch.Tensor | None,
        mask: torch.Tensor | None,
        batch: int,
        n: int,
        meta: tuple[str, tuple[int, ...]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if coords is None:
            coords = _default_coords(meta, n, device=device, dtype=dtype)
            coords = coords.unsqueeze(0).expand(batch, -1, -1)
        else:
            if coords.dim() == 2:
                coords = coords.unsqueeze(0).expand(batch, -1, -1)
            if coords.shape != (batch, n, 3):
                raise ValueError(
                    f"coords must have shape {(batch, n, 3)}, got {tuple(coords.shape)}"
                )
            coords = coords.to(device=device, dtype=dtype)
        if mask is None:
            mask = torch.ones(batch, n, device=device, dtype=torch.bool)
        else:
            if mask.shape != (batch, n):
                raise ValueError(f"mask must have shape {(batch, n)}, got {tuple(mask.shape)}")
            mask = mask.to(device=device, dtype=torch.bool)
        return coords, mask

    def balance_loss(self) -> torch.Tensor:
        """Switch-style importance/load penalty from the latest forward pass."""
        return self._balance_loss

    def routing_stats(self) -> dict[str, list[float]]:
        return {
            "importance": self.last_importance.detach().cpu().tolist(),
            "load": self.last_load.detach().cpu().tolist(),
            "temperature": [float(self.log_temperature.exp().detach().cpu())],
        }

    def forward(
        self,
        x: torch.Tensor,
        *,
        coords: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens, meta = _tokenize(x)
        batch, n, channels = tokens.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")
        coords, mask = self._aux(
            coords, mask, batch, n, meta, tokens.device, tokens.dtype
        )
        valid = mask[:, :, None].to(tokens.dtype)
        features = self.input(torch.cat((tokens, coords), dim=-1))
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (features * valid).sum(dim=1, keepdim=True) / denom
        pooled = self.context(pooled)
        routed = torch.cat((features, pooled.expand(-1, n, -1)), dim=-1)
        temperature = self.log_temperature.exp().clamp(0.25, 4.0)
        logits = self.router(routed) / temperature
        # Keep a dense probability path for the auxiliary loss. In particular,
        # top-1 routing otherwise turns importance into a non-differentiable
        # one-hot tensor and makes load balancing ineffective.
        dense_gates = torch.softmax(logits, dim=-1) * valid
        selection = torch.zeros_like(logits)
        if self.top_k < self.num_experts:
            top_values, top_indices = logits.topk(self.top_k, dim=-1)
            sparse = torch.full_like(logits, float("-inf"))
            routed_logits = sparse.scatter(-1, top_indices, top_values)
            selection.scatter_(-1, top_indices, 1.0)
            gates = torch.softmax(routed_logits, dim=-1) * valid
        else:
            selection.fill_(1.0)
            gates = dense_gates

        expert_values = torch.stack([expert(features) for expert in self.experts], dim=2)
        routed_value = torch.einsum("bne,bneh->bnh", gates, expert_values)
        mixed = self.shared(features) + routed_value
        if self.global_expert is not None:
            mixed = mixed + self.global_expert(pooled).expand(-1, n, -1)
        delta = self.output(mixed) * valid

        importance = dense_gates.sum(dim=(0, 1)) / valid.sum().clamp_min(1.0)
        selected = selection * valid
        load = selected.sum(dim=(0, 1)) / (
            valid.sum().clamp_min(1.0) * self.top_k
        )
        self._balance_loss = self.num_experts * (importance * load).sum()
        self.last_importance.copy_(importance.detach())
        self.last_load.copy_(load.detach())
        return _restore(tokens + delta, meta)
