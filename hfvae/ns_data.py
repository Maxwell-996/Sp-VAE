"""Read-only Navier-Stokes shard dataset compatible with diff_pde data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


DEFAULT_DIFFPDE_DATA = Path("/media/home/lizhonghao/diff_pde/data")


def resolve_data_root(data_root: str | os.PathLike[str] | None) -> Path:
    if data_root is not None:
        return Path(data_root).expanduser().resolve()
    if DEFAULT_DIFFPDE_DATA.exists():
        return DEFAULT_DIFFPDE_DATA
    return Path("data").resolve()


class NSShardDataset(Dataset):
    """Dataset for data/{split}/regime*.pt shards.

    Shard files are only read. Returned samples are normalized tensors in Wan
    layout: (3, T, H, W).
    """

    def __init__(
        self,
        data_root: str | os.PathLike[str] | None,
        split: str,
        *,
        t_clip: int = 49,
        norm_stats: tuple[list[float], list[float]] | None = None,
        in_memory: bool = False,
        max_shards: int | None = None,
        max_samples_per_shard: int | None = None,
    ):
        self.data_root = resolve_data_root(data_root)
        self.split = split
        self.split_dir = self.data_root / split
        man_path = self.split_dir / "manifest.pt"
        if not man_path.is_file():
            raise FileNotFoundError(f"missing manifest: {man_path}")
        self.manifest = torch.load(man_path, map_location="cpu", weights_only=False)
        shards = list(self.manifest["shards"])
        if max_shards is not None:
            shards = shards[:max_shards]
        self.shards = shards
        self.t_clip = int(t_clip)
        self.frames = int(self.manifest["frames"])
        self.in_memory = bool(in_memory)
        self.max_samples_per_shard = max_samples_per_shard
        self.index: list[tuple[int, int]] = []
        for si, sh in enumerate(self.shards):
            n = int(sh["n"])
            if max_samples_per_shard is not None:
                n = min(n, int(max_samples_per_shard))
            self.index.extend((si, li) for li in range(n))

        self._cache: dict[int, dict[str, Any]] = {}
        if self.in_memory:
            for si in range(len(self.shards)):
                self._cache[si] = self._load_shard(si)

        if norm_stats is None:
            stats_path = self.data_root / "norm_stats.pt"
            if stats_path.is_file():
                stats = torch.load(stats_path, map_location="cpu", weights_only=False)
                self.mean = list(map(float, stats["mean"]))
                self.std = list(map(float, stats["std"]))
            else:
                self.mean, self.std = self._compute_stats()
        else:
            self.mean, self.std = norm_stats
        self._mean_t = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1, 1)
        self._inv_std = (1.0 / (torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1, 1) + 1e-8))

    def _load_shard(self, si: int) -> dict[str, Any]:
        sh = self.shards[si]
        path = self.split_dir / sh["path"]
        d = torch.load(path, map_location="cpu", weights_only=False)
        d["uvp"] = d["uvp"].float()
        return d

    def _get_shard(self, si: int) -> dict[str, Any]:
        if si not in self._cache:
            self._cache = {si: self._load_shard(si)}
        return self._cache[si]

    def _compute_stats(self) -> tuple[list[float], list[float]]:
        csum = torch.zeros(3, dtype=torch.float64)
        csq = torch.zeros(3, dtype=torch.float64)
        count = 0
        for si in range(len(self.shards)):
            uvp = self._get_shard(si)["uvp"]
            for c in range(3):
                ch = uvp[:, c].double()
                csum[c] += ch.sum()
                csq[c] += ch.square().sum()
            count += uvp.shape[0] * uvp.shape[2] * uvp.shape[3] * uvp.shape[4]
        mean = csum / count
        std = (csq / count - mean.square()).clamp_min(0).sqrt()
        return mean.tolist(), std.tolist()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        si, li = self.index[idx]
        x = self._get_shard(si)["uvp"][li]
        if self.frames > self.t_clip:
            start = torch.randint(0, self.frames - self.t_clip + 1, (1,)).item()
        else:
            start = 0
        x = x[..., start:start + self.t_clip]
        x = (x - self._mean_t) * self._inv_std
        return x.permute(0, 3, 1, 2).contiguous()


class ShardContiguousSampler(Sampler[int]):
    """Sampler that minimizes shard re-loads in lazy mode."""

    def __init__(self, dataset: NSShardDataset, shuffle: bool = True):
        self.shuffle = bool(shuffle)
        self.by_shard: dict[int, list[int]] = {}
        for gi, (si, _li) in enumerate(dataset.index):
            self.by_shard.setdefault(si, []).append(gi)
        self.n = len(dataset.index)

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        shard_ids = list(self.by_shard)
        if self.shuffle:
            shard_ids = [shard_ids[i] for i in torch.randperm(len(shard_ids))]
        for si in shard_ids:
            idxs = self.by_shard[si]
            if self.shuffle:
                idxs = [idxs[i] for i in torch.randperm(len(idxs))]
            yield from idxs


def resize_video(x: torch.Tensor, out_hw: int | None, out_t: int | None) -> torch.Tensor:
    """Resize (B,C,T,H,W) with trilinear interpolation."""
    if out_hw is None and out_t is None:
        return x
    t = out_t if out_t is not None else x.shape[2]
    h = out_hw if out_hw is not None else x.shape[-2]
    w = out_hw if out_hw is not None else x.shape[-1]
    return F.interpolate(x, size=(t, h, w), mode="trilinear", align_corners=False)
