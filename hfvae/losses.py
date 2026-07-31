"""Composable VAE objectives for frequency-robust reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .frequency import (
    frequency_weighted_mse,
    high_frequency_leakage_loss,
    highpass_filter_2d,
    lowpass_filter_2d,
    spectral_energy_loss,
)


@dataclass
class HFVAELossConfig:
    beta_kl: float = 1e-4
    lambda_lowpass: float = 0.0
    lambda_hf: float = 0.0
    lambda_spectral: float = 0.0
    lambda_sobolev: float = 0.0
    lambda_latent_consistency: float = 0.0
    lowpass_cutoff: float = 0.35
    highpass_cutoff: float = 0.35
    transition: float = 0.08
    sobolev_s: float = 1.0


def kl_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    mu = mu.float()
    log_var = log_var.float()
    return -0.5 * torch.mean(1.0 + log_var - mu.square() - log_var.exp())


def base_reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(recon, target) + F.mse_loss(recon, target)


def latent_consistency_target(
    x: torch.Tensor,
    *,
    cutoff: float,
    noise_std: float = 0.05,
    transition: float = 0.08,
) -> torch.Tensor:
    noise = highpass_filter_2d(torch.randn_like(x), cutoff=cutoff, transition=transition)
    return x + float(noise_std) * noise


def compute_vae_loss(
    model,
    x: torch.Tensor,
    cfg: HFVAELossConfig,
    *,
    amp_reparameterize: bool = True,
    latent_consistency_noise: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    mu, log_var = model.encode_dist(x, [0, 1])
    log_var = log_var.clamp(-30, 20)
    z = model.reparameterize(mu, log_var) if amp_reparameterize else mu
    recon = model.decode(z, [0, 1])

    rec = base_reconstruction_loss(recon, x)
    kl = kl_loss(mu, log_var)
    loss = rec + cfg.beta_kl * kl
    parts = {"rec": float(rec.detach()), "kl": float(kl.detach())}

    if cfg.lambda_lowpass:
        lp = base_reconstruction_loss(
            lowpass_filter_2d(recon, cfg.lowpass_cutoff, cfg.transition),
            lowpass_filter_2d(x, cfg.lowpass_cutoff, cfg.transition),
        )
        loss = loss + cfg.lambda_lowpass * lp
        parts["lowpass"] = float(lp.detach())

    if cfg.lambda_hf:
        hf = high_frequency_leakage_loss(
            recon,
            x,
            cutoff=cfg.highpass_cutoff,
            transition=cfg.transition,
        )
        loss = loss + cfg.lambda_hf * hf
        parts["hf"] = float(hf.detach())

    if cfg.lambda_spectral:
        spec = spectral_energy_loss(recon, x)
        loss = loss + cfg.lambda_spectral * spec
        parts["spectral"] = float(spec.detach())

    if cfg.lambda_sobolev:
        sob = frequency_weighted_mse(
            recon,
            x,
            sobolev_s=cfg.sobolev_s,
            cutoff=cfg.highpass_cutoff,
            transition=cfg.transition,
        )
        loss = loss + cfg.lambda_sobolev * sob
        parts["sobolev"] = float(sob.detach())

    if cfg.lambda_latent_consistency:
        x_aug = latent_consistency_target(
            x,
            cutoff=cfg.highpass_cutoff,
            noise_std=latent_consistency_noise,
            transition=cfg.transition,
        )
        mu_aug, _ = model.encode_dist(x_aug, [0, 1])
        cons = F.mse_loss(mu_aug.float(), mu.detach().float())
        loss = loss + cfg.lambda_latent_consistency * cons
        parts["latent_consistency"] = float(cons.detach())

    parts["loss"] = float(loss.detach())
    return loss, parts
