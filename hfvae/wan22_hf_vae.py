"""High-frequency robust variants of the Wan2.2 VAE."""

from __future__ import annotations

import torch

from .frequency import (
    AdaptiveSpectralFilter2d,
    BlurPool2d,
    FrequencyBandMoE2d,
    HighFrequencyScaling2d,
)
from .wan22_imports import CausalConv3d, WanVAE_, patchify, unpatchify


class WanVAEHF(WanVAE_):
    """Wan2.2 VAE with local, opt-in high-frequency robustness modules.

    The base architecture remains Wan2.2's causal 3D VAE. This wrapper adds:
    - configurable patch_size (1 keeps image channels, 2 matches Wan's 2x patchify);
    - optional anti-aliasing before patchify;
    - optional learnable latent high-frequency scaling;
    - optional learnable decoder-side spectral filtering.
    """

    def __init__(
        self,
        *args,
        patch_size: int = 2,
        input_channels: int = 3,
        anti_alias: bool = False,
        latent_hfs: bool = False,
        output_hfs: bool = False,
        decoder_spectral_filter: bool = False,
        filter_cutoff: float | None = None,
        decoder_filter_min_gain: float = 0.02,
        hfs_cutoff: float = 0.35,
        latent_band_moe: bool = False,
        moe_hidden: int = 32,
        moe_low_cutoff: float = 0.20,
        moe_high_cutoff: float = 0.50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if patch_size not in (1, 2):
            raise ValueError("patch_size must be 1 or 2")
        self.ps = int(patch_size)
        self.input_channels = int(input_channels)

        enc_in = self.input_channels * self.ps * self.ps
        if self.encoder.conv1.in_channels != enc_in:
            enc_out = self.encoder.conv1.out_channels
            self.encoder.conv1 = CausalConv3d(enc_in, enc_out, 3, padding=1)

        dec_out = self.input_channels * self.ps * self.ps
        if self.decoder.head[-1].out_channels != dec_out:
            dec_in = self.decoder.head[-1].in_channels
            self.decoder.head[-1] = CausalConv3d(dec_in, dec_out, 3, padding=1)

        self.encoder_antialias = (
            BlurPool2d(self.input_channels, boundary="circular") if anti_alias else None
        )
        self.latent_hfs = (
            HighFrequencyScaling2d(self.z_dim, cutoff=hfs_cutoff, init_high=1.0)
            if latent_hfs
            else None
        )
        self.latent_band_moe = (
            FrequencyBandMoE2d(
                self.z_dim,
                hidden=moe_hidden,
                low_cutoff=moe_low_cutoff,
                high_cutoff=moe_high_cutoff,
            )
            if latent_band_moe
            else None
        )
        self.output_hfs = (
            HighFrequencyScaling2d(self.input_channels, cutoff=hfs_cutoff, init_high=0.75)
            if output_hfs
            else None
        )
        self.decoder_spectral_filter = (
            AdaptiveSpectralFilter2d(
                cutoff=filter_cutoff,
                min_gain=decoder_filter_min_gain,
            )
            if decoder_spectral_filter
            else None
        )

    def _encode_moments(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.clear_cache()
        if self.encoder_antialias is not None:
            x = self.encoder_antialias(x)
        x = patchify(x, patch_size=self.ps)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        out = None
        for i in range(iter_):
            self._enc_conv_idx = [0]
            seg = x[:, :, :1] if i == 0 else x[:, :, 1 + 4 * (i - 1):1 + 4 * i]
            y = self.encoder(
                seg,
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            out = y if out is None else torch.cat([out, y], dim=2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        self.clear_cache()
        return mu, log_var

    def encode_dist(
        self, x: torch.Tensor, scale: list[float | torch.Tensor] | tuple = (0, 1)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_var = self._encode_moments(x)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            mu = (mu - scale[0]) * scale[1]
        if self.latent_hfs is not None:
            mu = self.latent_hfs(mu)
        if self.latent_band_moe is not None:
            mu = self.latent_band_moe(mu)
        return mu, log_var

    def encode(self, x: torch.Tensor, scale):
        mu, _ = self.encode_dist(x, scale)
        return mu

    def decode(self, z: torch.Tensor, scale):
        self.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            z = z / scale[1] + scale[0]

        x = self.conv2(z)
        out = None
        for i in range(z.shape[2]):
            self._conv_idx = [0]
            y = self.decoder(
                x[:, :, i:i + 1],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
                first_chunk=(i == 0),
            )
            out = y if out is None else torch.cat([out, y], dim=2)
        out = unpatchify(out, patch_size=self.ps)
        if self.output_hfs is not None:
            out = self.output_hfs(out)
        if self.decoder_spectral_filter is not None:
            out = self.decoder_spectral_filter(out)
        self.clear_cache()
        return out
