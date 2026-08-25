# Sp-VAE 项目交接文档

本文档面向在新环境中接手本项目的 Codex agent，记录当前仓库状态、冻结的 VAE 方案、可复现实验入口、已有证据和明确边界。

## 1. 项目定位

本项目研究面向 Navier--Stokes（NS）时空场压缩的 high-frequency robust VAE。核心问题不是单纯降低重建误差，而是：

1. 压缩表示是否抑制编码器/解码器引入的非物理高频伪影。
2. VAE latent 是否保留对后续自回归动力学有用的信息。
3. 在固定压缩率下，latent-space dynamics 是否能优于显空间 dynamics。

当前研究边界是 VAE 方案。FNO、Transolver、CNextU 和 operator MoE 代码仍保留用于历史下游验证，但不属于最终 VAE 架构。

## 2. 当前冻结状态

当前主分支为 hfvae-research，冻结 commit 为：

~~~text
e8e259e Freeze final HFVAE global-dense8 profile
~~~

当前 canonical profile：

~~~text
configs/final_hfvae_global_dense8.json
profile = final_hfvae_global_dense8
status = frozen_candidate
~~~

该 profile 是当前研究候选的代码冻结，不代表已经证明对所有数据集或所有 dynamics operator 全局最优。

## 3. 冻结 VAE 架构

~~~text
regular periodic NS field
    -> spatial BlurPool anti-alias
    -> Wan2.2 causal 3D encoder
    -> latent HFS
    -> latent-only global + dense Token-MoE
         hidden=64, experts=8, top-k=8
         shared experts=2, global context=true
         token_choice routing, router noise=0
    -> latent z: (B, 16, 13, 8, 8) for (B, 3, 49, 128, 128)
    -> Wan2.2 causal 3D decoder
    -> output HFS
    -> soft decoder spectral filter
    -> reconstruction
~~~

基础 Wan 参数：

- dim=96, dec_dim=96, z_dim=16
- dim_mult=[1,2,4,4]
- temperal_downsample=[true,true,false]
- patch_size=2, input_channels=3
- anti_alias=true
- latent_hfs=true, hfs_cutoff=0.15
- output_hfs=true
- decoder_spectral_filter=true
- filter_cutoff=0.15, decoder_filter_min_gain=0.02

明确关闭的模块：

- latent_band_moe、output_band_moe、output_learnable_moe
- local/adaptive-local/expert-local context
- coordinate-kernel experts、anisotropic kernel
- dynamic top-k、expert-choice routing
- temporal difference、vorticity、spectral、wavelet objectives
- decoder hard spectral filter

实现位置：

- hfvae/wan22_hf_vae.py: Wan wrapper 和前向顺序。
- hfvae/frequency.py: BlurPool、HFS、soft spectral filter 和频域 loss。
- hfvae/learnable_moe.py: format-agnostic learnable Token-MoE。
- hfvae/losses.py: 可组合 VAE objective。
- hfvae/final_profile.py: frozen profile 校验和 exact builder。

## 4. 冻结 objective

基础重建为 L1 + MSE，固定权重来自 configs/final_hfvae_global_dense8.json：

~~~text
beta_kl=1e-4
lambda_lowpass=0.5
lambda_hf=0.1
lambda_sobolev=0.05
lambda_latent_consistency=0.05
lambda_moe_balance=0.01
lambda_noisy_reconstruction=0
lambda_spectral=0
lambda_wavelet=0
lambda_temporal=0
lambda_vorticity=0
lowpass_cutoff=0.10, highpass_cutoff=0.15, transition=0.04
consistency_noise=0.10, sobolev_s=1.0
~~~

训练入口使用 AdamW（weight decay 1e-4）和 CosineAnnealingLR。profile 值会覆盖 argparse 默认值，并写入 checkpoint metadata。

## 5. 运行入口和依赖

根目录没有单独的 requirements.txt 或 pyproject.toml。需要准备兼容的 Python、PyTorch 和 CUDA 环境；Wan2.2 的依赖说明在外部 Wan2.2/requirements.txt 和 Wan2.2/pyproject.toml。

hfvae/wan22_imports.py 会动态加载：

~~~text
Wan2.2/wan/modules/vae2_2.py
~~~

本机 Wan2.2 是独立嵌套仓库，commit 为 42bf4cf，上游为 https://github.com/Wan-Video/Wan2.2.git。主仓库 .gitignore 有意排除 Wan2.2/。本机的 codex-autoresearch/（commit d65f7c6）也是独立仓库，包含自动研究规范，不在主项目 Git 历史中。

数据仓库 /media/home/lizhonghao/diff_pde 只读，当前数据根为 /media/home/lizhonghao/diff_pde/data_128。不要修改该仓库。

### 最小验证

~~~bash
python -m unittest discover -s tests -v
python -m py_compile hfvae/final_profile.py experiments/train_hf_vae.py
~~~

测试验证冻结模型可构造，输入 (1,3,49,128,128) 的 latent 为 (1,16,13,8,8)，重建形状与输入一致，并确认 output MoE 和 band MoE 未启用。

### 冻结候选训练

~~~bash
python experiments/train_hf_vae.py --config configs/final_hfvae_global_dense8.json
~~~

profile 默认记录原实验初始化 checkpoint：runs/vae_hf_noisy0/hf_noisy0_best.pt。新环境没有该文件时，需单独转移 checkpoint 或明确改变初始化配置；改变后不再是原始冻结 run。

重建评估：

~~~bash
python experiments/eval_hf_vae.py --ckpt runs/final_hfvae_global_dense8/final_hfvae_global_dense8_best.pt --data-root /media/home/lizhonghao/diff_pde/data_128 --split valid --out-csv runs/final_hfvae_global_dense8/eval_valid.csv
~~~

latent cache 使用 experiments/cache_vae_latents.py；下游 operator 训练使用 experiments/train_latent_operator.py；解码物理 rollout 使用 experiments/eval_operator_physical.py。

## 6. 已有结果证据

以下数字来自本机 runs/ 下的 CSV/config/log，以及 docs/high_frequency_robust_vae.md 的记录。runs/ 被 .gitignore 排除，所以原始 checkpoint、latent cache、CSV、JSON 和日志没有推送到 GitHub，需要单独迁移。

### 静态 validation

| variant | rel-L2 | low-band rel-L2 | predicted HF energy | 来源 |
| --- | ---: | ---: | ---: | --- |
| Original Wan checkpoint | 0.343190363 | 0.334412235 | 3.50344612e-3 | runs/real_baseline_valid.csv |
| vanilla_ft | 0.148244058 | 0.138798359 | 1.20622218e-3 | runs/vanilla_ft_valid_full.csv |
| hf_noisy0 | 0.110946032 | 0.108807046 | 2.11378311e-5 | runs/hf_noisy0_valid_full.csv |
| global+dense8 | 0.057676648 | 0.055451684 | 1.23472634e-5 | runs/hf_ddp_global8/eval_valid.csv |

相对 vanilla_ft，global+dense8 的 validation rel-L2 约改善 61.1%。这只是静态重建证据，不能单独证明 latent dynamics 最优。

### 早期 downstream evidence

hf_noisy0 相对 vanilla_ft 的 latent test rollout：

| operator | vanilla_ft | hf_noisy0 |
| --- | ---: | ---: |
| FNO | 0.27161 | 0.24868 |
| Transolver | 0.46760 | 0.45967 |
| CNextU | 0.20241 | 0.18914 |

decoded physical-field rollout：

| operator | vanilla_ft | hf_noisy0 |
| --- | ---: | ---: |
| FNO | 0.26567 | 0.21907 |
| Transolver | 0.38050 | 0.34167 |
| CNextU | 0.22731 | 0.18303 |

来源：docs/high_frequency_robust_vae.md 的 Latent autoregression 和 Decoded physical-field rollout 小节；原始 artifacts 在本机 runs/。后续 global+dense8 矩阵中 latent rollout 与 decoded physical rollout 不总是同步改善，因此新 agent 必须同时报告三类指标。

## 7. irregular 数据边界

hfvae/learnable_moe.py 支持 (B,N,C)、(B,C,H,W)、(B,C,T,H,W)，并可接收 coords 和 mask；所以 MoE adapter 具有 arbitrary-token 接口。

但是冻结 Wan2.2 VAE 主干仍要求规则视频张量，encoder/decoder 使用规则空间卷积和 patchify。当前不能声称已经完成 arbitrary irregular-field VAE 压缩。正确结论是：Token-MoE 组件支持不规则 token，但真实 irregular VAE encoder/decoder 压缩与重建仍未完成验证；hfvae/irregular.py 主要是采样/评估辅助。

## 8. 边界和不可做事项

1. 不要修改 /media/home/lizhonghao/diff_pde。
2. 不要删除 hfvae/wan22_hf_vae.py 中的历史 opt-in 开关；它们用于 provenance 和 ablation。
3. 新实验单独建立 profile/config 和 run 目录，不覆盖 frozen profile。
4. 不要把 operator MoE、temporal difference 或 ACT-style coordinate warp 叠加回冻结 VAE，除非用户明确开启新的研究分支。
5. 不要把 runs/、checkpoint、latent cache 或 autoresearch-results/ 自动加入主仓库。
6. 新结论必须保存配置、seed、训练日志和评估输出，并在文档中标注来源。

## 9. 推荐接手顺序

1. 克隆 Sp-VAE，补齐 Wan2.2 到 commit 42bf4cf。
2. 准备 PyTorch/CUDA 环境和只读 diff_pde/data_128。
3. 运行最小 unittest，确认冻结 profile 和 Wan import 正常。
4. 转移或重新生成 hf_noisy0_best.pt，再训练/评估冻结候选。
5. 先复现 static VAE validation，再生成 latent cache 和 downstream rollout。
6. 只有复现基线后，才建立新的 ablation 分支；不要覆盖冻结 profile。

## 10. 未完成事项

- 迁移 checkpoint、latent cache、CSV、JSON 和 log 等独立实验 artifacts。
- 在相同 seed、split、训练预算下复现 global+dense8 的 static 和 downstream 指标。
- 确认 checkpoint 选择应以 latent dynamics 还是 decoded physical dynamics 为主。
- 完成真正 irregular input 的 VAE encoder/decoder 设计与评估。
- 为根目录补充可锁定版本的 environment/requirements 文件。

## 11. Git 发布信息

目标仓库：https://github.com/Maxwell-996/Sp-VAE

本仓库不保存私钥。专用私钥只应存在于本机，例如 /media/home/lizhonghao/id_ed25519_spvae。公钥需先添加到 GitHub 账户 SSH Keys，或作为具有写权限的 Deploy Key。
