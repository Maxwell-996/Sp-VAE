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
  --data-root /media/home/lizhonghao/diff_pde/data \
  --out-dir runs/hf_vae/hfs_lp \
  --tag hfs_lp \
  --epochs 100 --batch 2 --t-clip 49 --dim 96 --z-dim 16 \
  --anti-alias --latent-hfs --output-hfs --decoder-spectral-filter \
  --filter-cutoff 0.20 --hfs-cutoff 0.20 \
  --lambda-lowpass 0.5 --lambda-hf 0.05 --lambda-latent-consistency 0.05 \
  --lowpass-cutoff 0.10 --highpass-cutoff 0.20 --transition 0.04 \
  --amp --lazy

python experiments/eval_hf_vae.py \
  --ckpt runs/hf_vae/hfs_lp/hfs_lp_best.pt \
  --data-root /media/home/lizhonghao/diff_pde/data \
  --split valid --out-csv runs/hf_vae/hfs_lp/eval_valid.csv

python experiments/cache_vae_latents.py \
  --ckpt runs/hf_vae/hfs_lp/hfs_lp_best.pt \
  --data-root /media/home/lizhonghao/diff_pde/data \
  --split train --out-root runs/latents/hfs_lp

python experiments/cache_vae_latents.py \
  --ckpt runs/hf_vae/hfs_lp/hfs_lp_best.pt \
  --data-root /media/home/lizhonghao/diff_pde/data \
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

## 参考依据

- Deconvolution and Checkerboard Artifacts, Distill 2016: transposed convolution 容易产生棋盘伪影，resize-convolution 是常用替代。
- Making Convolutional Networks Shift-Invariant Again, ICML 2019: 下采样前 anti-aliasing/BlurPool 能改善平移稳定性。
- On the Spectral Bias of Neural Networks, ICML 2019: 常规网络存在低频先学/高频慢学的频率偏置。
- Sobolev Training for Neural Networks, NeurIPS 2017: 同时监督函数值和导数可编码更强目标信息。
- Fourier Neural Operator for Parametric PDEs, ICLR 2021: FNO 直接在 Fourier 模态上学习 PDE solution operator。
- Physics-Informed Neural Operator, 2021: 将数据 loss 和 PDE loss 结合，适合作为后续突破“重建好但残差高”的路径。
- Transolver, 2024: physics-attention 通过物理 slice/token 捕捉复杂几何或场相关性。
