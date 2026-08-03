# 高频鲁棒 VAE：数学形式与本仓库实验方案

## 问题定义

给定场 \(x\in\mathbb{R}^{C\times T\times H\times W}\)，VAE 为
\[
z\sim q_\phi(z|x),\qquad \hat{x}=D_\theta(z).
\]
普通目标
\[
\mathcal{L}_{base}=\|x-\hat{x}\|_1+\|x-\hat{x}\|_2^2+\beta\,\mathrm{KL}(q_\phi(z|x)\|p(z))
\]
只约束像素均值误差。对 NS 这种 PDE，误差进入 \(\nabla,\nabla^2\) 后在谱域分别被 \(k,k^2\) 放大，所以很小的网格高频伪影也会造成残差爆炸。

## 可测试的鲁棒形式

1. **带限重建**
\[
\mathcal{L}_{lp}=\|P_{\le k_c}(\hat{x}-x)\|_1+\|P_{\le k_c}(\hat{x}-x)\|_2^2.
\]
它把监督集中在动力学主能量频段，不让训练为不可辨别的高频噪声牺牲低频结构。

2. **高频泄漏惩罚**
\[
\mathcal{L}_{hf}=\|P_{>k_c}(\hat{x}-x)\|_2^2
\quad\text{或在光滑流中}\quad
\|P_{>k_c}\hat{x}\|_2^2.
\]
如果数据谱显示 \(k>k_c\) 近似为零，后一种是更强的去伪影先验。

3. **谱能量匹配**
\[
\mathcal{L}_{spec}=\|\log(|\mathcal{F}\hat{x}|^2+\epsilon)-\log(|\mathcal{F}x|^2+\epsilon)\|_2^2.
\]
它不强制相位完全一致，但能约束能量级联形状。

4. **带限 Sobolev 损失**
\[
\mathcal{L}_{H^s}=\sum_k (1+\|k\|^2)^s\,|\mathcal{F}(\hat{x}-x)(k)|^2.
\]
对 PDE 有用，但必须带限或平滑权重；否则会把训练导向噪声拟合。

5. **latent 一致性**
\[
\mathcal{L}_{cons}=\|E_\phi(x)-E_\phi(x+\eta_{hf})\|_2^2,
\quad \eta_{hf}=P_{>k_c}\eta.
\]
这是最直接服务“embedding 能否让自回归更稳”的约束：高频扰动不应大幅移动 latent。

## 本仓库实现

- `hfvae/frequency.py`：低通/高通、频域 MSE、谱能量损失、BlurPool、HFS、动态谱滤波。
- `hfvae/wan22_hf_vae.py`：在本地 Wan2.2 VAE 上加 anti-alias、latent HFS、output HFS、decoder spectral filter。
- `experiments/train_hf_vae.py`：只读 NS shard 数据，训练改进 VAE。
- `experiments/eval_hf_vae.py`：重建 relL2、低频 relL2、高频 relL2、高频能量。
- `experiments/score_robust_vae.py`：固定种子/预算的 clean + 高频扰动恢复 benchmark，结果只写临时目录。
- `experiments/cache_vae_latents.py`：把 VAE embedding 缓存在本目录。
- `experiments/train_latent_operator.py`：用同一 latent 数据测试 FNO、Transolver、CNextU 的自回归。

## 推荐第一组实验

```bash
python experiments/train_hf_vae.py \
  --data-root /media/home/lizhonghao/diff_pde/data_128 \
  --out-dir runs/hf_vae/hfs_lp \
  --tag hfs_lp \
  --epochs 100 --batch 2 --t-clip 49 --dim 96 --z-dim 16 \
  --anti-alias --latent-hfs --output-hfs --decoder-spectral-filter \
  --filter-cutoff 0.15 --hfs-cutoff 0.15 \
  --lambda-lowpass 0.5 --lambda-hf 0.10 --lambda-latent-consistency 0.05 \
  --lambda-noisy-reconstruction 0 \
  --lowpass-cutoff 0.10 --highpass-cutoff 0.15 --transition 0.04 \
  --amp --lazy

python experiments/eval_hf_vae.py \
  --ckpt runs/hf_vae/hfs_lp/hfs_lp_best.pt \
  --data-root /media/home/lizhonghao/diff_pde/data_128 \
  --split valid --out-csv runs/hf_vae/hfs_lp/eval_valid.csv

python experiments/cache_vae_latents.py \
  --ckpt runs/hf_vae/hfs_lp/hfs_lp_best.pt \
  --data-root /media/home/lizhonghao/diff_pde/data_128 \
  --split train --out-root runs/latents/hfs_lp

python experiments/cache_vae_latents.py \
  --ckpt runs/hf_vae/hfs_lp/hfs_lp_best.pt \
  --data-root /media/home/lizhonghao/diff_pde/data_128 \
  --split valid --out-root runs/latents/hfs_lp

python experiments/train_latent_operator.py \
  --latent-root runs/latents/hfs_lp \
  --act-root /media/home/lizhonghao/act/ACT-NeuralOP \
  --models fno transolver cnextu \
  --out-csv runs/latent_operator/hfs_lp/results.csv \
  --epochs 50 --history 4
```

对照组建议把 `--anti-alias --latent-hfs --output-hfs --decoder-spectral-filter` 和额外 loss 全部关掉，只保留原始重建 + KL。

## 核心 insight

1. **不要把“高频鲁棒”理解成盲目保留所有高频**。对当前 NS 数据，真实高频能量很低，高频鲁棒首先是“不要让编码器/解码器制造不可物理解释的高频”。因此 `lambda_hf` 和 decoder spectral filter 是第一优先级。

2. **PDE 残差不是重建误差的线性 proxy**。NS 残差含梯度和拉普拉斯，谱域会把误差乘上 \(k\) 或 \(k^2\)。同样 relL2 下，高频误差比低频误差更危险；但低频相位/幅值扰动也会破坏方程项抵消，所以 `lambda_lowpass` 不能省。

3. **embedding 的目标应该是“带限稳定 + 动力学充分”**。latent consistency 让 \(E(x)\approx E(x+\eta_{hf})\)，operator rollout 再检验这个压缩表示是否保留自回归所需的信息。

4. **模型侧与目标侧要一起做**。单靠 loss 会和上采样/下采样伪影对抗；所以实现里同时给了 anti-alias、latent/output HFS、decoder spectral filter 和频域 loss。

5. **下游三架构比较的解释**：FNO 有强低频/谱截断先验；Transolver 更依赖 token attention 捕捉物理相关；CNextU 有局部卷积归纳偏置。若鲁棒 VAE embedding 真的更好，应在 rollout_rel 上至少不输 raw/普通 VAE，并且在 FNO 与 Transolver 上更明显。

## 实验结果（data_128）

所有结果使用 `/media/home/lizhonghao/diff_pde/data_128` 的 6 个 regime，train/valid/test 分别为 1200/240/240 条轨迹，VAE 输入为 `49x128x128`，latent 为 `(16,13,8,8)`。`vanilla_ft` 和 `hf_noisy0` 都从同一个 Wan checkpoint 开始，在同一数据上训练 1 epoch；算子固定 history=4、hidden=64、30 epochs、seed=3407，并在独立 test 上评估。

### VAE valid

| variant | rel-L2 | low-band rel-L2 | predicted HF energy |
| --- | ---: | ---: | ---: |
| original Wan checkpoint | 0.34319 | 0.33441 | 3.50e-3 |
| vanilla_ft: reconstruction + KL | 0.14824 | 0.13880 | 1.21e-3 |
| hf_noisy0: HF objective, no noisy reconstruction | **0.11095** | **0.10881** | **2.11e-5** |

因此，数据域适配本身解释了大部分 rel-L2 改善；HF 方法的独立收益主要是把 decoder 产生的非物理高频能量压低约两个数量级。`lambda_noisy_reconstruction=0` 是关键消融：加入它后 valid rel-L2 回到 0.14821。

继续从 `hf_noisy0` checkpoint 训练 2 个 epoch 后，静态重建还能明显改善：`hf_noisy0_3ep` 的 valid rel-L2 为 **0.06529**，low-band rel-L2 为 **0.06305**，predicted HF energy 为 `1.10e-5`。但更好的静态重建不自动等于更好的 latent 动力学，见下文的 3 epoch operator 结果。

### 频谱目标与结构消融

| variant | valid rel-L2 | predicted HF energy | 结论 |
| --- | ---: | ---: | --- |
| hf_noisy0 | **0.11095** | 2.11e-5 | 1 epoch 默认 |
| log spectral energy, weight=0.001 | 0.13604 | - | 逐频 log-energy matching 过度约束 |
| latent consistency weight=0.10 | 0.15532 | 2.07e-5 | 不改善 clean 或 robust 指标 |
| HF leakage weight=0.20 | 0.13762 | 1.65e-5 | 惩罚过强，牺牲低频重建 |
| Sobolev order `s=2` | 0.13353 | 2.43e-5 | 高阶导数约束不稳定 |
| consistency noise=0.20 | 0.13189 | 2.54e-5 | 更强扰动训练不占优 |
| all cutoffs=0.10 | 0.11269 | **7.75e-7** | HF energy 最低，但 clean 略差 |
| decoder hard filter (`min_gain=0`) | 0.12543 | 1.71e-5 | 硬带限反而变差 |

去掉完整 HF 配置中的单个结构模块后，三种 operator 的 test rollout 均不如完整模型，尤其是 output HFS；decoder filter 对 Transolver/CNextU 的长期稳定性也很重要。逐频能量 matching 没有带来收益，说明当前 NS 数据更需要“抑制非物理高频泄漏”，而不是逐频复制幅值谱。

### Latent autoregression

| operator | vanilla_ft one-step | hf_noisy0 one-step | vanilla_ft rollout | hf_noisy0 rollout |
| --- | ---: | ---: | ---: | ---: |
| FNO | 0.05253 | **0.04694** | 0.27161 | **0.24868** |
| Transolver | 0.06321 | **0.05634** | 0.46760 | **0.45967** |
| CNextU | 0.04481 | **0.04104** | 0.20241 | **0.18914** |

HF embedding 在三个模型的 one-step 和 rollout 都优于同预算 vanilla_ft。加入 temporal difference loss 后，FNO 的 rollout 进一步改善（latent 0.23706，decoded physical field 0.20151），但 Transolver rollout 恶化到 0.52659；所以 temporal 项适合做 FNO 专用配置，不应作为通用默认项。

`hf_noisy0_3ep` 的 latent test rollout 为 FNO `0.26210`、Transolver `0.56439`、CNextU `0.22063`，均不如 1 epoch `hf_noisy0`。这表明 operator 看到的不是单纯的重建误差：继续训练改变了 latent 的动力学尺度/可预测性，必须把 VAE 与 operator 联合评估。

### Decoded physical-field rollout

在 240 条 test trajectories 上，把递归 latent 预测完整解码回 `(u,v,p)`：

| operator | vanilla_ft | hf_noisy0 | relative change |
| --- | ---: | ---: | ---: |
| FNO | 0.26567 | **0.21907** | -17.5% |
| Transolver | 0.38050 | **0.34167** | -10.2% |
| CNextU | 0.22731 | **0.18303** | -19.5% |

这验证了 latent rollout 的收益会传到物理场，而不是只改善归一化 embedding 的数值。

3 epoch checkpoint 解码后的 physical rollout 为 FNO `0.19911`、Transolver `0.41185`、CNextU `0.17968`。因此继续训练对 FNO/CNextU 的真实场误差仍有帮助，但会损害 Transolver；当前通用推荐仍是 1 epoch `hf_noisy0`，而 3 epoch 版本更适合作为 FNO/CNextU 的候选配置。

### 研究结论

1. 对当前 NS 数据，“高频鲁棒”首先是抑制 encoder/decoder 伪影，而不是尽量保存所有高频；数据本身在归一化频率 0.15 以上的能量已很小。
2. 普通重建误差不是 NS rollout 的充分目标：它没有区分低频动力学误差和高频伪影，且 noisy reconstruction 会与 clean reconstruction 竞争。
3. 当前推荐默认是 anti-alias + latent/output spectral scaling + decoder spectral filter，加 `lambda_lowpass=0.5`、`lambda_hf=0.10`、`lambda_sobolev=0.05`、`lambda_latent_consistency=0.05`，并关闭 noisy reconstruction。
4. 结论依赖下游架构：1 epoch `hf_noisy0` 对 FNO、Transolver、CNextU 都有收益；继续训练或 temporal difference 都可能改变不同 operator 的排序。
5. 选 VAE 不能只看 valid reconstruction：当前通用 latent rollout 最佳是 1 epoch `hf_noisy0`，clean reconstruction 最佳是 3 epoch `hf_noisy0_3ep`；最终应按目标 operator 选择 checkpoint。

`experiments/eval_operator_physical.py` 用对应 VAE 将完整 latent rollout 解码后评估物理场误差；`runs/` 下的 checkpoints、latent caches 和 CSV/JSON 是本次实验的原始产物，均未修改 `diff_pde`。

## 参考依据

- Deconvolution and Checkerboard Artifacts, Distill 2016: transposed convolution 容易产生棋盘伪影，resize-convolution 是常用替代。
- [Making Convolutional Networks Shift-Invariant Again, ICML 2019](https://proceedings.mlr.press/v97/zhang19a.html): 下采样前 anti-aliasing/BlurPool 能改善平移稳定性。
- On the Spectral Bias of Neural Networks, ICML 2019: 常规网络存在低频先学/高频慢学的频率偏置。
- [Sobolev Training for Neural Networks, NeurIPS 2017](https://arxiv.org/abs/1706.04859): 同时监督函数值和导数可编码更强目标信息。
- [Fourier Neural Operator for Parametric PDEs, ICLR 2021](https://arxiv.org/abs/2010.08895): FNO 直接在 Fourier 模态上学习 PDE solution operator。
- Physics-Informed Neural Operator, 2021: 将数据 loss 和 PDE loss 结合，适合作为后续突破“重建好但残差高”的路径。
- Transolver, 2024: physics-attention 通过物理 slice/token 捕捉复杂几何或场相关性。
