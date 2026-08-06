"""Format-agnostic sampling and periodic lifting for latent field observations."""

from __future__ import annotations

import math

import torch


def periodic_grid_coordinates(
    height: int, width: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return periodic ``(x, y)`` coordinates for a regular query grid."""
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) / height,
        torch.arange(width, device=device, dtype=dtype) / width,
        indexing="ij",
    )
    return torch.stack((xx, yy), dim=-1).reshape(height * width, 2)


def sample_irregular_coordinates(
    batch: int,
    count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
    pattern: str = "uniform",
) -> torch.Tensor:
    """Sample continuous periodic coordinates without assuming a mesh layout."""
    if batch < 1 or count < 1:
        raise ValueError("batch and count must be positive")
    if pattern not in {"uniform", "clustered", "boundary"}:
        raise ValueError("pattern must be uniform, clustered, or boundary")
    coords = torch.rand(
        batch, count, 2, device=device, dtype=dtype, generator=generator
    )
    if pattern == "clustered":
        centers = torch.rand(batch, 1, 2, device=device, dtype=dtype, generator=generator)
        offsets = torch.randn(
            batch, count, 2, device=device, dtype=dtype, generator=generator
        ) * 0.12
        coords = (centers + offsets).remainder(1.0)
    elif pattern == "boundary":
        edge = torch.randint(0, 4, (batch, count), device=device, generator=generator)
        distance = torch.rand(
            batch, count, device=device, dtype=dtype, generator=generator
        ) * 0.08
        along = torch.rand(
            batch, count, device=device, dtype=dtype, generator=generator
        )
        coords = torch.empty(batch, count, 2, device=device, dtype=dtype)
        coords[..., 0] = torch.where(
            edge == 0,
            distance,
            torch.where(edge == 1, 1.0 - distance, along),
        )
        coords[..., 1] = torch.where(
            edge == 2,
            distance,
            torch.where(edge == 3, 1.0 - distance, along),
        )
    return coords


def sample_periodic_bilinear(
    field: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Sample ``(B,C,H,W)`` fields at continuous periodic coordinates."""
    if field.dim() != 4 or coords.dim() != 3 or coords.shape[-1] != 2:
        raise ValueError("field must be (B,C,H,W) and coords must be (B,N,2)")
    batch, channels, height, width = field.shape
    if coords.shape[0] != batch:
        raise ValueError("field and coordinate batch dimensions must match")
    if mask is not None and mask.shape != coords.shape[:2]:
        raise ValueError("mask must be (B,N)")
    coords = coords.to(device=field.device, dtype=field.dtype).remainder(1.0)
    x = coords[..., 0] * width
    y = coords[..., 1] * height
    x0 = x.floor().long().remainder(width)
    y0 = y.floor().long().remainder(height)
    x1 = (x0 + 1).remainder(width)
    y1 = (y0 + 1).remainder(height)
    wx = (x - x.floor()).unsqueeze(1)
    wy = (y - y.floor()).unsqueeze(1)
    flat = field.reshape(batch, channels, height * width)

    def gather(row: torch.Tensor, column: torch.Tensor) -> torch.Tensor:
        index = (row * width + column).unsqueeze(1).expand(-1, channels, -1)
        return flat.gather(2, index)

    top_left = gather(y0, x0)
    top_right = gather(y0, x1)
    bottom_left = gather(y1, x0)
    bottom_right = gather(y1, x1)
    values = (
        (1.0 - wy) * ((1.0 - wx) * top_left + wx * top_right)
        + wy * ((1.0 - wx) * bottom_left + wx * bottom_right)
    )
    if mask is not None:
        values = values * mask[:, None, :].to(values.dtype)
    return values


def periodic_rbf_lift(
    values: torch.Tensor,
    coords: torch.Tensor,
    *,
    height: int,
    width: int,
    length_scale: float,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift arbitrary sampled values onto a periodic grid with normalized RBFs.

    The returned coverage is a coordinate-only confidence map, useful for
    distinguishing a true zero observation from an unobserved location.
    """
    if values.dim() != 3 or coords.shape != (values.shape[0], values.shape[2], 2):
        raise ValueError("values must be (B,C,N) and coords must be (B,N,2)")
    if length_scale <= 0:
        raise ValueError("length_scale must be positive")
    batch, _channels, count = values.shape
    if mask is None:
        mask = torch.ones(batch, count, device=values.device, dtype=torch.bool)
    elif mask.shape != (batch, count):
        raise ValueError("mask must be (B,N)")
    grid = periodic_grid_coordinates(
        height, width, device=values.device, dtype=values.dtype
    )
    delta = (coords.to(device=values.device, dtype=values.dtype).remainder(1.0)[:, :, None]
        - grid[None, None])
    delta = delta.abs()
    delta = torch.minimum(delta, 1.0 - delta)
    distance2 = delta.square().sum(dim=-1)
    weights = torch.exp(-distance2 / (2.0 * length_scale * length_scale))
    weights = weights * mask[:, :, None].to(weights.dtype)
    coverage = weights.sum(dim=1)
    normalized = weights / coverage[:, None].clamp_min(1e-8)
    lifted = torch.einsum("bnq,bcn->bcq", normalized, values)
    return lifted.reshape(batch, values.shape[1], height, width), coverage.reshape(
        batch, 1, height, width
    )


def default_rbf_length_scale(count: int) -> float:
    """Stable length scale based only on observation density in a unit torus."""
    if count < 1:
        raise ValueError("count must be positive")
    return 1.25 / math.sqrt(count)
