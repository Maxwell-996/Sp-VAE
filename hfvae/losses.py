"""Composable VAE objectives for frequency-robust reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .frequency import (
    frequency_weighted_mse,
    high_frequency_leakage_loss,
    haar_detail_reconstruction_loss,
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
    lambda_wavelet: float = 0.0
    lambda_sobolev: float = 0.0
    lambda_latent_consistency: float = 0.0
    lambda_noisy_reconstruction: float = 0.0
    lambda_temporal: float = 0.0
    lambda_vorticity: float = 0.0
    lambda_moe_balance: float = 0.0
    wavelet_levels: int = 2
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


def temporal_difference_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match first temporal differences so compression preserves dynamics."""
    if recon.shape[2] < 2:
        return recon.new_zeros(())
    recon_dt = recon[:, :, 1:] - recon[:, :, :-1]
    target_dt = target[:, :, 1:] - target[:, :, :-1]
    return F.l1_loss(recon_dt, target_dt) + F.mse_loss(recon_dt, target_dt)


def vorticity_reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Match periodic-grid vorticity for the two velocity channels."""
    if recon.shape[1] < 2 or target.shape[1] < 2:
        raise ValueError("vorticity loss requires u and v channels")
    recon = recon.float()
    target = target.float()

    def curl_z(field: torch.Tensor) -> torch.Tensor:
        u, v = field[:, 0], field[:, 1]
        du_dy = 0.5 * (torch.roll(u, -1, dims=-2) - torch.roll(u, 1, dims=-2))
        dv_dx = 0.5 * (torch.roll(v, -1, dims=-1) - torch.roll(v, 1, dims=-1))
        return dv_dx - du_dy

    return F.l1_loss(curl_z(recon), curl_z(target)) + F.mse_loss(
        curl_z(recon), curl_z(target)
    )


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

    if cfg.lambda_wavelet:
        wavelet = haar_detail_reconstruction_loss(
            recon,
            x,
            levels=cfg.wavelet_levels,
        )
        loss = loss + cfg.lambda_wavelet * wavelet
        parts["wavelet"] = float(wavelet.detach())

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

    if cfg.lambda_temporal:
        temporal = temporal_difference_loss(recon, x)
        loss = loss + cfg.lambda_temporal * temporal
        parts["temporal"] = float(temporal.detach())

    if cfg.lambda_vorticity:
        vorticity = vorticity_reconstruction_loss(recon, x)
        loss = loss + cfg.lambda_vorticity * vorticity
        parts["vorticity"] = float(vorticity.detach())

    x_aug = None
    mu_aug = None
    if cfg.lambda_latent_consistency or cfg.lambda_noisy_reconstruction:
        x_aug = latent_consistency_target(
            x,
            cutoff=cfg.highpass_cutoff,
            noise_std=latent_consistency_noise,
            transition=cfg.transition,
        )
        mu_aug, _ = model.encode_dist(x_aug, [0, 1])

    if cfg.lambda_noisy_reconstruction:
        recon_aug = model.decode(mu_aug, [0, 1])
        noisy_rec = base_reconstruction_loss(recon_aug, x)
        loss = loss + cfg.lambda_noisy_reconstruction * noisy_rec
        parts["noisy_reconstruction"] = float(noisy_rec.detach())

    if cfg.lambda_latent_consistency:
        cons = F.mse_loss(mu_aug.float(), mu.detach().float())
        loss = loss + cfg.lambda_latent_consistency * cons
        parts["latent_consistency"] = float(cons.detach())

    if cfg.lambda_moe_balance:
        balance = model.moe_balance_loss()
        loss = loss + cfg.lambda_moe_balance * balance
        parts["moe_balance"] = float(balance.detach())

    parts["loss"] = float(loss.detach())
    return loss, parts
