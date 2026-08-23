from __future__ import annotations

import unittest

import torch

from experiments.train_hf_vae import apply_profile_config
from hfvae.final_profile import (
    FINAL_PROFILE_NAME,
    build_final_model,
    final_model_config,
    load_profile,
    validate_final_profile,
)


class FinalProfileTest(unittest.TestCase):
    def test_training_entry_uses_profile_values(self) -> None:
        args = type("Args", (), {"config": "configs/final_hfvae_global_dense8.json"})()
        apply_profile_config(args)
        self.assertEqual(args.profile, FINAL_PROFILE_NAME)
        self.assertEqual(args.dim, 96)
        self.assertEqual(args.learnable_moe_experts, 8)
        self.assertEqual(args.learnable_moe_top_k, 8)
        self.assertEqual(args.learnable_moe_shared_experts, 2)
        self.assertTrue(args.latent_learnable_moe)
        self.assertFalse(args.output_learnable_moe)
        self.assertEqual(args.lambda_hf, 0.1)
        self.assertEqual(args.lambda_noisy_reconstruction, 0.0)

    def test_profile_is_canonical(self) -> None:
        profile = load_profile()
        self.assertEqual(profile["profile"], FINAL_PROFILE_NAME)
        validate_final_profile(profile)
        self.assertEqual(profile["model"], final_model_config())
        self.assertFalse(profile["model"]["output_learnable_moe"])
        self.assertFalse(profile["model"]["output_band_moe"])
        self.assertFalse(profile["model"]["latent_band_moe"])

    def test_model_constructs_and_preserves_shapes(self) -> None:
        torch.manual_seed(3407)
        model = build_final_model().eval()
        self.assertIsNotNone(model.encoder_antialias)
        self.assertIsNotNone(model.latent_hfs)
        self.assertIsNotNone(model.output_hfs)
        self.assertIsNotNone(model.decoder_spectral_filter)
        self.assertIsNotNone(model.latent_learnable_moe)
        self.assertIsNone(model.output_learnable_moe)
        self.assertIsNone(model.latent_band_moe)
        self.assertIsNone(model.output_band_moe)
        self.assertEqual(model.latent_learnable_moe.num_experts, 8)
        self.assertEqual(model.latent_learnable_moe.top_k, 8)
        self.assertEqual(model.latent_learnable_moe.shared_experts, 2)
        self.assertTrue(model.latent_learnable_moe.global_context_enabled)

        x = torch.randn(1, 3, 49, 128, 128)
        with torch.no_grad():
            mu, log_var = model.encode_dist(x, [0, 1])
            recon = model.decode(mu, [0, 1])
        self.assertEqual(tuple(mu.shape), (1, 16, 13, 8, 8))
        self.assertEqual(tuple(log_var.shape), (1, 16, 13, 8, 8))
        self.assertEqual(tuple(recon.shape), tuple(x.shape))


if __name__ == "__main__":
    unittest.main()
