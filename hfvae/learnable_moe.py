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

    def forward(
        self,
        x: torch.Tensor,
        *,
        coords: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.net(x)


class _CoordinateKernelExpert(nn.Module):
    """Content- and coordinate-conditioned kernel operator for arbitrary tokens."""

    def __init__(
        self,
        hidden: int,
        expansion: int = 2,
        adaptive_radius: bool = False,
    ):
        super().__init__()
        attention_hidden = max(8, hidden // 2)
        self.query_norm = nn.LayerNorm(hidden)
        self.key_norm = nn.LayerNorm(hidden)
        self.query = nn.Linear(hidden, attention_hidden, bias=False)
        self.key = nn.Linear(hidden, attention_hidden, bias=False)
        self.value = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.coordinate_bias = nn.Sequential(
            nn.Linear(3, attention_hidden),
            nn.GELU(),
            nn.Linear(attention_hidden, 1),
        )
        # Begin with an isotropic learned-radius kernel. The residual adapter
        # remains an exact identity because LearnableTokenMoE.output is zero.
        nn.init.zeros_(self.coordinate_bias[-1].weight)
        nn.init.zeros_(self.coordinate_bias[-1].bias)
        self.log_length_scale = nn.Parameter(torch.tensor(math.log(0.25)))
        self.radius_router = (
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
            if adaptive_radius
            else None
        )
        if self.radius_router is not None:
            nn.init.zeros_(self.radius_router[-1].weight)
            nn.init.zeros_(self.radius_router[-1].bias)
        self.register_buffer(
            "last_effective_length_scale",
            torch.tensor(0.25),
            persistent=False,
        )
        self.register_buffer(
            "last_effective_length_scale_std",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_effective_length_scale_min",
            torch.tensor(0.25),
            persistent=False,
        )
        self.register_buffer(
            "last_effective_length_scale_max",
            torch.tensor(0.25),
            persistent=False,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * expansion),
            nn.GELU(),
            nn.Linear(hidden * expansion, hidden),
        )

    def length_scale(self) -> torch.Tensor:
        return self.log_length_scale.exp().clamp(1e-3, 4.0)

    def effective_length_scale(self) -> torch.Tensor:
        return self.last_effective_length_scale

    def effective_length_scale_stats(self) -> list[float]:
        return [
            float(self.last_effective_length_scale.detach().cpu()),
            float(self.last_effective_length_scale_std.detach().cpu()),
            float(self.last_effective_length_scale_min.detach().cpu()),
            float(self.last_effective_length_scale_max.detach().cpu()),
        ]

    def forward(
        self,
        x: torch.Tensor,
        *,
        coords: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if coords is None or mask is None:
            raise ValueError("coordinate kernel experts require coords and mask")
        query = self.query(self.query_norm(x))
        key = self.key(self.key_norm(x))
        content_logits = torch.einsum("bih,bjh->bij", query, key)
        content_logits = content_logits / math.sqrt(query.shape[-1])

        delta = coords[:, :, None, :].float() - coords[:, None, :, :].float()
        distance2 = delta.square().sum(dim=-1)
        scale = self.length_scale().to(dtype=x.dtype)
        if self.radius_router is not None:
            valid = mask[:, :, None].to(x.dtype)
            pooled = (x * valid).sum(dim=1, keepdim=True) / valid.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            # Zero-initialized modulation preserves the fixed-kernel baseline.
            modulation = 0.5 * torch.tanh(self.radius_router(pooled))
            scale = (scale * modulation.exp()).clamp(1e-3, 4.0)
        scale_stats = scale.detach().float().reshape(-1)
        self.last_effective_length_scale.copy_(scale_stats.mean())
        self.last_effective_length_scale_std.copy_(scale_stats.std(unbiased=False))
        self.last_effective_length_scale_min.copy_(scale_stats.min())
        self.last_effective_length_scale_max.copy_(scale_stats.max())
        coordinate_logits = self.coordinate_bias(delta.to(x.dtype)).squeeze(-1)
        logits = content_logits - distance2.to(x.dtype) / (2.0 * scale.square())
        logits = logits + coordinate_logits
        key_valid = mask[:, None, :]
        logits = logits.masked_fill(~key_valid, torch.finfo(logits.dtype).min)
        has_valid_key = mask.any(dim=1, keepdim=True).unsqueeze(-1)
        safe_logits = torch.where(has_valid_key, logits, torch.zeros_like(logits))
        weights = torch.softmax(safe_logits, dim=-1) * mask[:, :, None].to(x.dtype)
        message = torch.einsum("bij,bjh->bih", weights, self.value(x))
        return self.output(x + message)


class LearnableTokenMoE(nn.Module):
    """Shared-plus-routed token MoE with learnable sparse routing.

    Inputs can be ``(B,N,C)``, ``(B,C,H,W)``, or ``(B,C,T,H,W)``. For point
    clouds or irregular meshes, callers can provide ``coords=(B,N,3)`` and a
    boolean ``mask=(B,N)`` without resampling to a regular grid. The residual
    projection is zero initialized so a newly attached adapter is initially an
    identity map. ``token_choice`` gives every token up to ``top_k`` experts;
    ``expert_choice`` lets every expert choose a fixed-capacity set of tokens,
    while the output is still written back at the original token positions.
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
        adaptive_global: bool = False,
        router_inhibition: bool = False,
        local_context: bool = False,
        adaptive_local_mix: bool = False,
        expert_local_mix: bool = False,
        routing: str = "token_choice",
        expert_kind: str = "mlp",
        shared_experts: int = 1,
        router_noise_std: float = 0.0,
    ):
        super().__init__()
        if channels < 1 or hidden < 1:
            raise ValueError("channels and hidden must be positive")
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")
        if routing not in {"token_choice", "expert_choice"}:
            raise ValueError("routing must be token_choice or expert_choice")
        if expert_kind not in {"mlp", "kernel", "adaptive_kernel"}:
            raise ValueError("expert_kind must be mlp, kernel, or adaptive_kernel")
        if shared_experts < 1:
            raise ValueError("shared_experts must be positive")
        if router_noise_std < 0:
            raise ValueError("router_noise_std must be non-negative")
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.routing = routing
        self.expert_kind = expert_kind
        self.shared_experts = int(shared_experts)
        self.router_noise_std = float(router_noise_std)
        self.global_context_enabled = bool(global_context)
        self.adaptive_global = bool(adaptive_global)
        self.router_inhibition = bool(router_inhibition)
        self.local_context_enabled = bool(local_context)
        self.adaptive_local_mix = bool(local_context and adaptive_local_mix)
        self.expert_local_mix = bool(local_context and expert_local_mix)
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
        self.router = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, num_experts),
        )
        if expert_kind == "mlp":
            experts = [_Expert(hidden, expansion=expansion) for _ in range(num_experts)]
        else:
            experts = [
                _CoordinateKernelExpert(
                    hidden,
                    expansion=expansion,
                    adaptive_radius=expert_kind == "adaptive_kernel",
                )
                for _ in range(num_experts)
            ]
        self.experts = nn.ModuleList(experts)
        self.shared = _Expert(hidden, expansion=expansion)
        self.shared_extra = nn.ModuleList(
            [_Expert(hidden, expansion=expansion) for _ in range(shared_experts - 1)]
        )
        self.global_expert = _Expert(hidden, expansion=expansion) if global_context else None
        self.global_gate = (
            nn.Sequential(
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, 1),
            )
            if global_context and adaptive_global
            else None
        )
        if self.global_gate is not None:
            nn.init.zeros_(self.global_gate[-1].weight)
            nn.init.zeros_(self.global_gate[-1].bias)
        self.local_log_scales = (
            nn.Parameter(torch.log(torch.tensor([0.05, 0.15, 0.40])))
            if local_context
            else None
        )
        self.local_router = (
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
            if local_context
            else None
        )
        self.local_expert = (
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
            if local_context
            else None
        )
        self.local_scale_mixer = (
            nn.Sequential(
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, 3),
            )
            if self.adaptive_local_mix
            else None
        )
        self.expert_local_mix_logits = (
            nn.Parameter(torch.zeros(num_experts, 3)) if self.expert_local_mix else None
        )
        self.expert_local = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
                for _ in range(num_experts)
            ]
            if self.expert_local_mix
            else []
        )
        for adapter in (self.local_router, self.local_expert, self.local_scale_mixer):
            if adapter is not None:
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
        for adapter in self.expert_local:
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
        self.output = nn.Linear(hidden, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.5)))
        self._balance_loss = torch.zeros(())
        self._last_global_gate = 0.5
        self.register_buffer(
            "last_importance", torch.full((num_experts,), 1.0 / num_experts)
        )
        self.register_buffer("last_load", torch.full((num_experts,), 1.0 / num_experts))
        self.register_buffer(
            "last_local_mix",
            torch.full((3,), 1.0 / 3.0),
            persistent=False,
        )
        self.register_buffer(
            "last_local_mix_std",
            torch.zeros(3),
            persistent=False,
        )
        self.register_buffer(
            "last_expert_local_mix",
            torch.full((num_experts, 3), 1.0 / 3.0),
            persistent=False,
        )

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
            "global_gate": [float(self._last_global_gate)],
            "local_scales": (
                self.local_log_scales.detach().exp().cpu().tolist()
                if self.local_log_scales is not None
                else []
            ),
            "local_mix": (
                self.last_local_mix.detach().cpu().tolist()
                if self.local_scale_mixer is not None
                else []
            ),
            "local_mix_std": (
                self.last_local_mix_std.detach().cpu().tolist()
                if self.local_scale_mixer is not None
                else []
            ),
            "expert_local_mix": (
                self.last_expert_local_mix.detach().cpu().tolist()
                if self.expert_local_mix_logits is not None
                else []
            ),
            "kernel_scales": (
                [float(expert.effective_length_scale().detach().cpu()) for expert in self.experts]
                if self.expert_kind in {"kernel", "adaptive_kernel"}
                else []
            ),
            "kernel_scale_stats": (
                [expert.effective_length_scale_stats() for expert in self.experts]
                if self.expert_kind in {"kernel", "adaptive_kernel"}
                else []
            ),
        }

    def _local_delta(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        pooled_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate learnable-radius coordinate neighborhoods without reshaping."""
        coords_f = coords.float()
        distance2 = torch.cdist(coords_f, coords_f).square()
        pair_valid = (mask[:, :, None] & mask[:, None, :]).unsqueeze(1)
        scales = self.local_log_scales.exp().clamp(1e-3, 2.0)
        weights = torch.exp(
            -distance2.unsqueeze(1) / (2.0 * scales[None, :, None, None].square())
        ) * pair_valid
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        neighborhood = torch.einsum("bsij,bjh->bsih", weights, features.float())
        deltas = neighborhood - features.float().unsqueeze(1)
        if self.local_scale_mixer is None:
            return deltas.mean(dim=1).to(features.dtype), deltas.to(features.dtype)

        local_logits = self.local_scale_mixer(
            torch.cat((features, pooled_tokens), dim=-1)
        )
        local_mix = torch.softmax(local_logits, dim=-1) * mask[:, :, None]
        valid_mix = local_mix[mask]
        if valid_mix.numel():
            self.last_local_mix.copy_(valid_mix.detach().float().mean(dim=0))
            self.last_local_mix_std.copy_(valid_mix.detach().float().std(dim=0, unbiased=False))
        return (
            torch.einsum("bns,bsnh->bnh", local_mix, deltas).to(features.dtype),
            deltas.to(features.dtype),
        )

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
        pooled_tokens = pooled.expand(-1, n, -1)
        local_delta, local_deltas = (
            self._local_delta(features, coords, mask, pooled_tokens)
            if self.local_context_enabled
            else (None, None)
        )
        expert_features = (
            features + self.local_expert(local_delta)
            if local_delta is not None and self.local_expert is not None
            else features
        )
        router_features = (
            features + self.local_router(local_delta)
            if local_delta is not None and self.local_router is not None
            else features
        )
        router_local = router_features - pooled_tokens if self.router_inhibition else router_features
        routed = torch.cat((router_local, pooled_tokens), dim=-1)
        temperature = self.log_temperature.exp().clamp(0.25, 4.0)
        logits = self.router(routed) / temperature
        if self.training and self.router_noise_std:
            logits = logits + torch.randn_like(logits) * self.router_noise_std
        # Keep a dense probability path for the auxiliary loss. In particular,
        # top-1 routing otherwise turns importance into a non-differentiable
        # one-hot tensor and makes load balancing ineffective.
        dense_gates = torch.softmax(logits, dim=-1) * valid
        selection = torch.zeros_like(logits)
        if self.routing == "expert_choice":
            # Equal expert capacity removes token-side winner-take-all
            # pressure. The top_k value sets the target average number of
            # expert assignments per token, matching token-choice compute.
            flat_logits = logits.reshape(batch * n, self.num_experts)
            flat_valid = mask.reshape(batch * n)
            valid_count = int(flat_valid.sum().item())
            capacity = max(
                1,
                math.ceil(self.top_k * max(valid_count, 1) / self.num_experts),
            )
            capacity = min(capacity, batch * n)
            choice_logits = flat_logits.masked_fill(~flat_valid[:, None], float("-inf"))
            _, top_indices = choice_logits.topk(capacity, dim=0)
            selection_flat = torch.zeros_like(flat_logits)
            selection_flat.scatter_(0, top_indices, 1.0)
            selection = selection_flat.reshape(batch, n, self.num_experts) * valid
            routed_logits = logits.masked_fill(selection == 0, float("-inf"))
            has_route = selection.any(dim=-1, keepdim=True)
            safe_logits = torch.where(has_route, routed_logits, torch.zeros_like(routed_logits))
            route_gates = torch.softmax(safe_logits, dim=-1)
            gates = route_gates * has_route * valid
        else:
            if self.top_k < self.num_experts:
                top_values, top_indices = logits.topk(self.top_k, dim=-1)
                sparse = torch.full_like(logits, float("-inf"))
                routed_logits = sparse.scatter(-1, top_indices, top_values)
                selection.scatter_(-1, top_indices, 1.0)
                gates = torch.softmax(routed_logits, dim=-1) * valid
            else:
                selection.fill_(1.0)
                gates = dense_gates

        if self.expert_local_mix_logits is not None and local_deltas is not None:
            expert_local_mix = torch.softmax(self.expert_local_mix_logits, dim=-1)
            self.last_expert_local_mix.copy_(expert_local_mix.detach())
            expert_deltas = torch.einsum("es,bsnh->benh", expert_local_mix, local_deltas)
            expert_inputs = [
                expert_features + adapter(expert_deltas[:, index])
                for index, adapter in enumerate(self.expert_local)
            ]
        else:
            expert_inputs = [expert_features] * self.num_experts
        expert_values = torch.stack(
            [
                expert(expert_input, coords=coords, mask=mask)
                for expert, expert_input in zip(self.experts, expert_inputs)
            ],
            dim=2,
        )
        routed_value = torch.einsum("bne,bneh->bnh", gates, expert_values)
        shared_values = [self.shared(expert_features)]
        shared_values.extend(expert(expert_features) for expert in self.shared_extra)
        shared_value = torch.stack(shared_values, dim=2).mean(dim=2)
        mixed = shared_value + routed_value
        if self.global_expert is not None:
            global_value = self.global_expert(pooled).expand(-1, n, -1)
            if self.global_gate is not None:
                global_gate = torch.sigmoid(self.global_gate(routed))
                mixed = shared_value + (1.0 - global_gate) * routed_value + global_gate * global_value
                self._last_global_gate = float((global_gate * valid).sum().detach() / valid.sum().clamp_min(1.0))
            else:
                mixed = mixed + global_value
        delta = self.output(mixed) * valid

        importance = dense_gates.sum(dim=(0, 1)) / valid.sum().clamp_min(1.0)
        selected = selection * valid
        if self.routing == "expert_choice":
            load = selected.sum(dim=(0, 1)) / selected.sum().clamp_min(1.0)
        else:
            load = selected.sum(dim=(0, 1)) / (
                valid.sum().clamp_min(1.0) * self.top_k
            )
        self._balance_loss = self.num_experts * (importance * load).sum()
        self.last_importance.copy_(importance.detach())
        self.last_load.copy_(load.detach())
        return _restore(tokens + delta, meta)
