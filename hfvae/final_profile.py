"""Frozen VAE profile used for the current HFVAE candidate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .wan22_hf_vae import WanVAEHF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_PROFILE_PATH = PROJECT_ROOT / "configs" / "final_hfvae_global_dense8.json"
FINAL_PROFILE_NAME = "final_hfvae_global_dense8"


_FINAL_MODEL = {
    "dim": 96,
    "dec_dim": 96,
    "z_dim": 16,
    "dim_mult": [1, 2, 4, 4],
    "temperal_downsample": [True, True, False],
    "patch_size": 2,
    "input_channels": 3,
    "anti_alias": True,
    "latent_hfs": True,
    "output_hfs": True,
    "decoder_spectral_filter": True,
    "filter_cutoff": 0.15,
    "decoder_filter_min_gain": 0.02,
    "hfs_cutoff": 0.15,
    "latent_band_moe": False,
    "moe_hidden": 32,
    "moe_low_cutoff": 0.2,
    "moe_high_cutoff": 0.5,
    "moe_global_low": False,
    "moe_decomposition": "fft",
    "moe_cross_band": False,
    "moe_shared_expert": False,
    "moe_router_smooth": False,
    "latent_learnable_moe": True,
    "output_learnable_moe": False,
    "learnable_moe_hidden": 64,
    "learnable_moe_experts": 8,
    "learnable_moe_top_k": 8,
    "learnable_moe_global": True,
    "learnable_moe_adaptive_global": False,
    "learnable_moe_router_inhibition": False,
    "learnable_moe_local_context": False,
    "learnable_moe_adaptive_local_mix": False,
    "learnable_moe_expert_local_mix": False,
    "learnable_moe_routing": "token_choice",
    "learnable_moe_shared_experts": 2,
    "learnable_moe_router_noise": 0.0,
    "output_band_moe": False,
}

_FINAL_LOSS = {
    "beta_kl": 0.0001,
    "lambda_lowpass": 0.5,
    "lambda_hf": 0.1,
    "lambda_spectral": 0.0,
    "lambda_wavelet": 0.0,
    "lambda_sobolev": 0.05,
    "lambda_latent_consistency": 0.05,
    "lambda_noisy_reconstruction": 0.0,
    "lambda_temporal": 0.0,
    "lambda_vorticity": 0.0,
    "lambda_moe_balance": 0.01,
    "wavelet_levels": 2,
    "lowpass_cutoff": 0.1,
    "highpass_cutoff": 0.15,
    "transition": 0.04,
    "sobolev_s": 1.0,
    "consistency_noise": 0.1,
}


def load_profile(path: str | Path = FINAL_PROFILE_PATH) -> dict[str, Any]:
    """Load a JSON profile and validate the frozen profile when requested."""
    profile_path = Path(path)
    with profile_path.open(encoding="utf-8") as handle:
        profile = json.load(handle)
    if profile.get("profile") == FINAL_PROFILE_NAME:
        validate_final_profile(profile)
    return profile


def flatten_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Map structured profile sections to the argparse namespace fields."""
    values: dict[str, Any] = {}
    for section in ("model", "loss", "training", "data"):
        section_values = profile.get(section, {})
        if not isinstance(section_values, dict):
            raise ValueError(f"profile section {section!r} must be an object")
        values.update(section_values)
    return values


def validate_final_profile(profile: dict[str, Any]) -> None:
    """Reject silent architecture or objective drift from the frozen candidate."""
    if profile.get("profile") != FINAL_PROFILE_NAME:
        raise ValueError(
            f"expected profile {FINAL_PROFILE_NAME!r}, got {profile.get('profile')!r}"
        )
    if profile.get("version") != 1:
        raise ValueError("final HFVAE profile version must be 1")
    if profile.get("model") != _FINAL_MODEL:
        raise ValueError("final HFVAE model configuration does not match the frozen profile")
    if profile.get("loss") != _FINAL_LOSS:
        raise ValueError("final HFVAE loss configuration does not match the frozen profile")


def build_final_model(
    profile: dict[str, Any] | None = None,
) -> WanVAEHF:
    """Construct exactly the frozen VAE architecture."""
    profile = load_profile() if profile is None else profile
    validate_final_profile(profile)
    return WanVAEHF(**dict(profile["model"]))


def final_model_config() -> dict[str, Any]:
    """Return a copy of the canonical model configuration for tests/tools."""
    return dict(_FINAL_MODEL)


def final_loss_config() -> dict[str, Any]:
    """Return a copy of the canonical objective configuration for tests/tools."""
    return dict(_FINAL_LOSS)


__all__ = [
    "FINAL_PROFILE_NAME",
    "FINAL_PROFILE_PATH",
    "build_final_model",
    "final_loss_config",
    "final_model_config",
    "flatten_profile",
    "load_profile",
    "validate_final_profile",
]
