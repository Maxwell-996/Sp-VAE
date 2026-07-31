"""Frequency-domain operators and losses for robust reconstruction."""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_5d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Return x as (B,C,T,H,W) plus original shape."""
    orig = tuple(x.shape)
    if x.dim() == 5:
        return x, orig
    if x.dim() == 4:
        return x.unsqueeze(2), orig
    raise ValueError(f"expected 4D/5D tensor, got {tuple(x.shape)}")


def _restore_shape(x: torch.Tensor, orig_shape: tuple[int, ...]) -> torch.Tensor:
    if len(orig_shape) == 4:
        return x.squeeze(2)
    return x


def radial_frequency_grid(
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    normalized: bool = True,
) -> torch.Tensor:
    """Radial Fourier frequency grid for fft2 frequencies.

    If normalized=True, radius is divided by the Nyquist radius, so the
    diagonal Nyquist corner is near 1.0. Cutoffs can then be resolution agnostic.
    """
    ky = torch.fft.fftfreq(height, d=1.0, device=device).to(dtype)
    kx = torch.fft.fftfreq(width, d=1.0, device=device).to(dtype)
    yy, xx = torch.meshgrid(ky, kx, indexing="ij")
    rr = torch.sqrt(xx.square() + yy.square())
    if normalized:
        rr = rr / math.sqrt(0.5)
    return rr


def raised_cosine_lowpass(
    height: int,
    width: int,
    cutoff: float,
    transition: float = 0.08,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Smooth low-pass mask on normalized radial frequency."""
    rr = radial_frequency_grid(height, width, device=device, dtype=dtype)
    if transition <= 0:
        return (rr <= cutoff).to(dtype)
    lo = cutoff - transition / 2.0
    hi = cutoff + transition / 2.0
    mask = torch.ones_like(rr)
    mask = torch.where(rr >= hi, torch.zeros_like(mask), mask)
    mid = (rr > lo) & (rr < hi)
    mask[mid] = 0.5 * (1.0 + torch.cos(math.pi * (rr[mid] - lo) / transition))
    return mask


def fourier_filter_2d(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply a real-valued spatial Fourier mask over the final H,W dims."""
    x5, orig = _as_5d(x)
    dtype = x5.dtype
    mask = mask.to(device=x5.device, dtype=torch.float32)
    x_flat = x5.float().reshape(-1, x5.shape[-2], x5.shape[-1])
    fx = torch.fft.fft2(x_flat, norm="ortho")
    y = torch.fft.ifft2(fx * mask, norm="ortho").real
    y = y.reshape_as(x5).to(dtype)
    return _restore_shape(y, orig)


def lowpass_filter_2d(
    x: torch.Tensor,
    cutoff: float = 0.35,
    transition: float = 0.08,
) -> torch.Tensor:
    h, w = x.shape[-2:]
    mask = raised_cosine_lowpass(
        h, w, cutoff, transition, device=x.device, dtype=torch.float32
    )
    return fourier_filter_2d(x, mask)


def highpass_filter_2d(
    x: torch.Tensor,
    cutoff: float = 0.35,
    transition: float = 0.08,
) -> torch.Tensor:
    h, w = x.shape[-2:]
    mask = 1.0 - raised_cosine_lowpass(
        h, w, cutoff, transition, device=x.device, dtype=torch.float32
    )
    return fourier_filter_2d(x, mask)


def frequency_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sobolev_s: float = 0.0,
    high_boost: float = 0.0,
    cutoff: float = 0.35,
    transition: float = 0.08,
) -> torch.Tensor:
    """Fourier-domain MSE with Sobolev and high-band weighting.

    sobolev_s > 0 approximates an H^s loss. high_boost adds extra weight above
    cutoff. Keep cutoff/transition smooth to avoid ringing in the gradient.
    """
    err, _ = _as_5d(pred - target)
    h, w = err.shape[-2:]
    rr = radial_frequency_grid(h, w, device=err.device, dtype=torch.float32)
    weight = (1.0 + rr.square()).pow(float(sobolev_s))
    if high_boost:
        hp = 1.0 - raised_cosine_lowpass(
            h, w, cutoff, transition, device=err.device, dtype=torch.float32
        )
        weight = weight * (1.0 + float(high_boost) * hp)
    ef = torch.fft.fft2(err.float().reshape(-1, h, w), norm="ortho")
    return (ef.abs().square() * weight).mean()


def spectral_energy_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    log_energy: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Match per-frequency energy, independent of phase."""
    pred5, _ = _as_5d(pred)
    target5, _ = _as_5d(target)
    h, w = pred5.shape[-2:]
    pf = torch.fft.fft2(pred5.float().reshape(-1, h, w), norm="ortho")
    tf = torch.fft.fft2(target5.float().reshape(-1, h, w), norm="ortho")
    pe = pf.abs().square()
    te = tf.abs().square()
    if log_energy:
        pe = torch.log(pe + eps)
        te = torch.log(te + eps)
    return F.mse_loss(pe, te)


def high_frequency_leakage_loss(
    pred: torch.Tensor,
    target: torch.Tensor | None = None,
    *,
    cutoff: float = 0.35,
    transition: float = 0.08,
) -> torch.Tensor:
    """Penalize high-band energy.

    If target is provided, this penalizes high-band reconstruction error. If target
    is None, it penalizes high-band output energy directly, useful when the data
    manifold is known to be smooth.
    """
    hp = highpass_filter_2d(pred - target, cutoff, transition) if target is not None else highpass_filter_2d(pred, cutoff, transition)
    return hp.float().square().mean()


class BlurPool2d(nn.Module):
    """Fixed anti-aliasing blur over spatial H,W dimensions."""

    def __init__(self, channels: int, filt_size: int = 3, boundary: str = "circular"):
        super().__init__()
        if filt_size not in (3, 5):
            raise ValueError("filt_size must be 3 or 5")
        base = torch.tensor([1.0, 2.0, 1.0]) if filt_size == 3 else torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        kernel = torch.outer(base, base)
        kernel = kernel / kernel.sum()
        self.channels = int(channels)
        self.boundary = boundary
        self.pad = filt_size // 2
        self.register_buffer("kernel", kernel.view(1, 1, filt_size, filt_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x5, orig = _as_5d(x)
        b, c, t, h, w = x5.shape
        if c != self.channels:
            raise ValueError(f"BlurPool2d expected {self.channels} channels, got {c}")
        flat = x5.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        weight = self.kernel.to(dtype=flat.dtype).expand(c, 1, -1, -1)
        flat = F.pad(flat, (self.pad, self.pad, self.pad, self.pad), mode=self.boundary)
        y = F.conv2d(flat, weight, groups=c)
        y = y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return _restore_shape(y, orig)


class HighFrequencyScaling2d(nn.Module):
    """Learnable low/high split scaling in spatial frequency space."""

    def __init__(
        self,
        channels: int,
        cutoff: float = 0.35,
        transition: float = 0.08,
        init_low: float = 1.0,
        init_high: float = 1.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.cutoff = float(cutoff)
        self.transition = float(transition)
        self.log_low = nn.Parameter(torch.full((channels,), math.log(init_low)))
        self.log_high = nn.Parameter(torch.full((channels,), math.log(init_high)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x5, orig = _as_5d(x)
        low = lowpass_filter_2d(x5, self.cutoff, self.transition)
        high = x5 - low
        shape = (1, self.channels, 1, 1, 1)
        y = self.log_low.exp().view(shape) * low + self.log_high.exp().view(shape) * high
        return _restore_shape(y, orig)


class AdaptiveSpectralFilter2d(nn.Module):
    """Learnable radial spectral output filter with dynamic input resolution."""

    def __init__(
        self,
        max_radius: int = 512,
        cutoff: float | None = None,
        transition: float = 0.08,
        min_gain: float = 0.0,
        max_gain: float = 4.0,
    ):
        super().__init__()
        self.max_radius = int(max_radius)
        self.cutoff = cutoff
        self.transition = float(transition)
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        self.log_gain = nn.Parameter(torch.zeros(self.max_radius + 1))

    def _indices(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        ky = torch.fft.fftfreq(h, d=1.0 / h, device=device)
        kx = torch.fft.fftfreq(w, d=1.0 / w, device=device)
        yy, xx = torch.meshgrid(ky, kx, indexing="ij")
        return torch.sqrt(xx.square() + yy.square()).round().long().clamp(max=self.max_radius)

    def _base_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        if self.cutoff is None:
            return torch.ones(h, w, device=device)
        return raised_cosine_lowpass(
            h, w, self.cutoff, self.transition, device=device, dtype=torch.float32
        ).clamp_min(self.min_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x5, orig = _as_5d(x)
        dtype = x5.dtype
        h, w = x5.shape[-2:]
        idx = self._indices(h, w, x5.device)
        gain = self.log_gain.exp().clamp(max=self.max_gain)[idx]
        gain = gain * self._base_mask(h, w, x5.device)
        flat = x5.float().reshape(-1, h, w)
        y = torch.fft.ifft2(torch.fft.fft2(flat, norm="ortho") * gain, norm="ortho").real
        y = y.reshape_as(x5).to(dtype)
        return _restore_shape(y, orig)
