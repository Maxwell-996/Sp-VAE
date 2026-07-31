"""Local Wan2.2 module imports without touching package-level generation deps."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WAN_ROOT = ROOT / "Wan2.2"


def _load_module(name: str, relpath: str):
    path = WAN_ROOT / relpath
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find Wan2.2 module: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_vae22 = _load_module("hfvae_wan_vae22", "wan/modules/vae2_2.py")

WanVAE_ = _vae22.WanVAE_
CausalConv3d = _vae22.CausalConv3d
patchify = _vae22.patchify
unpatchify = _vae22.unpatchify
count_conv3d = _vae22.count_conv3d

__all__ = [
    "WanVAE_",
    "CausalConv3d",
    "patchify",
    "unpatchify",
    "count_conv3d",
]
