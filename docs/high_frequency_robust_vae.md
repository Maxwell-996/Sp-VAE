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

## 多频带 MoE 与 FFT 的边界

### 不是把频率硬切成三个独立模型

对规则、周期网格，可以写成

\[
x=P_lx+P_mx+P_hx,
\qquad
y=x+W_o\left[\sum_{b\in\{l,m,h\}}
\pi_b(c(x,r))E_b(P_bx)+E_{cross}(P_lx,P_mx,P_hx)\right].
\]

这里 `P_b` 是低、中、高频分解，`E_b` 是带有不同归纳偏置的专家，
`\pi_b` 是依赖位置 `r` 和局部物理状态的 soft router，`E_cross` 是跨频带
交互。最后一项不能省略：Navier--Stokes 的非线性对流项满足

\[
\widehat{(u\cdot\nabla)u}(k)
 =\sum_{p+q=k}(iq\cdot\hat u(p))\hat u(q),
\]

因此低频背景、中频结构和高频剪切层会发生 triadic coupling。完全独立的
low/mid/high experts 只适合做诊断，不适合直接作为动力学模型。当前实现中的
`--moe-cross-band` 正是一个零初始化的交互路径：初始仍是 Wan VAE，训练后才
允许专家交换信息。

MoE 解决的是容量和归纳偏置的分配，不会恢复采样前已经丢失的频率。对当前 NS
数据，更合理的专家职责是：

- low：全局谱或大感受野分支，保持慢变背景和长程压力耦合；
- mid：局部卷积/窗口注意力，处理涡旋尺度和中尺度输运；
- high：局部残差、Haar/wavelet 或梯度感知分支，处理剪切层和 resolved detail；
- cross：低频背景调制高频专家，避免高频专家独立地产生伪影；
- router：低频均值、高频能量、速度梯度、涡量和坐标/边界信息，而不是只读
  一个 FFT bin。

### FFT 模态数的理论上限

增加 FNO 的 modes 只能减少模型的谱截断误差，不能突破以下限制：

1. 采样频率超过 Nyquist 后的成分已经 alias 到低频，无法从样本唯一恢复；
2. 有限网格上的 FFT 是周期边界下的离散投影，边界不周期时会出现泄漏；
3. 卷积、非线性和下采样会产生未被解析的高频，增加 modes 不能补回 sub-grid
   信息；
4. PDE 导数在谱域乘以 `|k|` 或 `|k|^2`，小的高频幅值误差可能比较大的低频
   误差更快地放大为残差。

所以“保留更多 modes”与“高频鲁棒”不是同一个目标。VAE 中首先要抑制
decoder 产生的非物理高频，同时保留数据中真实且可预测的 resolved detail。

### 规则和不规则采样

标准 FFT 的正交、平移等变和 `O(N log N)` 优势依赖规则笛卡尔网格以及明确的
边界条件。对于随机观测点、非规则网格或复杂几何，直接把点值 reshape 成图像
再做 FFT 会把采样误差伪装成频率误差。可用三种替代：

1. 先用守恒/周期感知的 mesh-to-grid，再使用 FFT expert；
2. 用 point kernel、graph message passing 或 local attention 作为 high/mid
   experts；
3. 在几何域上用 learned coordinate deformation、图 Laplacian 或局部 wavelet
   basis 建立谱坐标，再进行 soft routing。

Geo-FNO 的 learned deformation 是“先把一般几何映射到可 FFT 空间”的代表；
Wavelet Neural Operator 用空间局部化的尺度函数替代纯全局 Fourier basis；M²M
则直接以多尺度、多专家和动态 gate 处理 PDE 中不同区域的振荡尺度。它们支持
当前的结论：频率/尺度应作为专家的归纳偏置，而不应被当成精确、全局且互不耦合
的真值标签。

## Learnable Token MoE

固定 `P_low/P_mid/P_high` 只适合作为消融基线。当前主方向改为保留 token 对应
关系的 learnable routed MoE。对每个 token `i`，先构造内容和坐标条件特征

\[
h_i=\operatorname{MLP}([x_i, c_i]),
\qquad
g_i=\operatorname{TopKSoftmax}(R([h_i,\bar h])),
\]

再用 shared expert 和 routed experts 混合：

\[
y_i=x_i+W_o\left(F_{shared}(h_i)+
\sum_{e=1}^{E}g_{i,e}F_e(h_i)\right).
\]

其中 `W_o` 零初始化，保证新模块接入时仍是 identity；`TopK` 只稀疏化专家
混合权重，不丢弃 token，也不改变 token 的空间/时间顺序。当前实现仍计算所有
专家以保持实现简单，后续再测真正的 capacity-routed sparse dispatch。`\bar h` 是 mask-aware
的全局上下文，不是 slot 或 pooling 后的唯一表示。为避免所有 token 塌缩到同一
专家，训练时可以加入 Switch-style 的 importance/load balance：

\[
\mathcal L_{balance}=E\sum_e
\operatorname{Importance}_e\operatorname{Load}_e.
\]

`Importance` 使用 top-k 截断前的 dense softmax 概率，因此 top-1 路由仍能得到
对 router 的可微平衡梯度；`Load` 使用截断后的 expert 选择，并按每个 token 的
`k` 个选择归一化。`top_k=1` 和 `top_k=2` 因而可以在相同的 balance 标度下比较。

该接口支持 `(B,N,C)` token、`(B,C,H,W)` image、`(B,C,T,H,W)` video，以及外部
`coords=(B,N,3)` 和 `mask=(B,N)`。因此不规则点集可以直接输入同一个 adapter，
不需要 reshape 到周期网格。坐标、局部状态和 latent content 共同决定专家权重，
专家的“低频/高频/边界/输运”功能由训练形成，而不是由 FFT cutoff 预先规定。

实现位于 `hfvae/learnable_moe.py`，Wan latent 入口由
`--latent-learnable-moe` 打开；`--output-learnable-moe` 用于 decoder-side
对照。当前保留配置使用 4 个专家的 dense token routing、2 个 shared experts、
一个 mask-aware global expert、router 输入 LayerNorm、初始 temperature `0.5`
以及训练期 noise `0`。专家职责不预先命名为 low/mid/high，也不把 slot 作为压缩
表示；它们只通过内容、坐标和有效采样 mask 学习数据相关的分工。

### Token-choice 与 expert-choice

当前实现同时支持两种仍保留 token 对应关系的可学习稀疏路由。`token_choice` 是
标准的 token-choice top-k：每个 token 选择最多 `k` 个 expert。它的优点是容量随
token 的需要变化，但当某些 token 的 router logit 偏大时，容易出现 expert
collapse。

`expert_choice` 反过来让每个 expert 从所有有效 token 中选择固定容量的 token。容量
设为 `ceil(k N / E)`，其中 `N` 是有效 token 数，`E` 是 expert 数；因此总的 expert
assignment 与 token-choice top-k 接近，但每个 expert 的 hard load 在结构上均匀。未
被任何 expert 选中的 token 仍经过 shared expert，选中的 expert 输出按 router logit
做 token-wise 聚合，最后写回原位置：

\[
  y_i = x_i + W_o\left(f_{shared}(h_i) +
  \sum_{e\in A_i} \operatorname{softmax}_{e\in A_i}(r_{ie})f_e(h_i)\right).
\]

这里的 `A_i` 不是固定频带，也不是 slot，而是由当前内容、坐标和全局上下文条件
动态产生的 expert 集合。该变体用于检验“expert 主动寻找局部高频模式”是否比
“token 竞争 expert”更稳定；它仍支持任意 token、coords 和 mask，不要求规则 FFT
网格。`--learnable-moe-routing expert_choice` 可打开该对照。

另一个行业中常见的稳定化方法是 noisy top-k gating：训练时在 router logit 上加入
小幅噪声，让早期尚未被选中的 expert 获得探索机会，测试时关闭噪声。对于
DeepSeekMoE 风格的 fine-grained specialization，则把 routed expert 切成更多小
expert，同时保留多个 shared expert 表达所有 token 都需要的公共变换。当前接口用
`--learnable-moe-router-noise` 和 `--learnable-moe-shared-experts` 暴露这两个变量，
仍然只产生 token-wise residual，不改变 latent token 数量或位置对应关系。

### 对“用 MoE 处理高、中、低频”的直接回答

可以使用 MoE，而且它比单一 FFT backbone 更适合表达 NS 中不同空间尺度的局部差异，
但 MoE 的作用是分配容量和归纳偏置，不是恢复采样前已经丢失的信息。给定采样算子
`S`，观测为 `y=Sx+epsilon` 时，任何模型都受限于不可辨识项

\[
\| (I-P_{\mathrm{resolved}})x\| + \|A_{\mathrm{alias}}x\| + \|\epsilon\|,
\]

其中第一项是未解析尺度，第二项是 aliasing，第三项是观测噪声。FFT modes 增大只
能减少模型自己的谱截断误差，不能消除这三项；非周期边界还会额外产生 spectral
leakage。非规则点集没有固定的正交 DFT 基，直接 reshape 后做 FFT 会把几何采样误差
混入频率误差。

因此推荐的主线 MoE 形式是

\[
\Delta z=W_o\left(\sum_b \pi_b(z,r,m)E_b(G_b(z,r,m))
 +E_{\mathrm{cross}}(z_l,z_m,z_h)\right),
\qquad z'=z+\Delta z,
\]

其中 `G_b` 不需要是固定的 Fourier 频带；它可以是共享 token MLP、局部消息传递、
坐标条件 kernel 或其他可学习 expert。对于当前规则 `data_128`，低/中/高频只应
作为分析标签而不是路由监督；对于 irregular 分支，point kernel、graph message
passing 或 token attention 可直接接收 `(value, coordinate, mask)`，不要求规则网格。

`W_o` 零初始化、soft/top-k gate 和 load-balance 正则很重要：新 adapter 的初始行为
应等于 Wan VAE，避免改变原有 latent 坐标系；训练后再逐步学习哪些区域和尺度需要
额外容量。仓库中的 `FrequencyBandMoE2d` 已覆盖规则网格的 FFT/local + cross-band
版本，`hfvae/learnable_moe.py` 则提供了支持坐标和 mask 的 token MoE；后者是当前
不规则采样和无预设频带主线的候选，但仍需通过 VAE 与下游 rollout 实验验证。

### 当前实验矩阵

在固定 `data_128`、相同 Wan 初始化、相同 1 epoch 训练和相同 operator 协议下，
依次比较：

| 变体 | 主要假设 | 后续判据 |
| --- | --- | --- |
| latent FFT MoE | 频带容量分配有效 | 静态重建和 latent rollout |
| latent FFT + cross-band | Transolver 变差来自频带耦合不足 | 尤其比较 Transolver rollout |
| latent local + cross-band | FFT 周期投影/分带误差是瓶颈 | 与 FFT + cross-band 配对比较 |
| output-band MoE | decoder 端分频能改善细节但少扰动 latent | static 与 decoded physical rollout |
| global-low MoE | 低频长程专家能降低背景误差 | FNO/Transolver/CNextU 三者均不能显著退化 |
| learnable token MoE top-2 | 内容/坐标条件路由比固定频带更适合 latent | static、三算子 rollout 和 arbitrary-token smoke test |
| learnable token MoE top-1/8 experts | 稀疏度与专家容量的 trade-off | 与 top-2/4 配对比较 |
| expert-choice token MoE | 固定 expert 容量抑制 token-choice collapse | 与 8-expert/top-2 配对比较 static、routing 和 rollout |
| noisy top-k / fine-grained shared MoE | 训练探索和 shared capacity 改善专家专门化 | 8-expert noisy 与 16-expert/4-route/2-shared 配对比较 |
| global + dense shared token MoE | 全局背景条件调制局部专家，同时保留 token 对应关系 | 4-expert dense/global/2-shared 的完整四卡确认 |
| adaptive global/local token MoE | 每个 token 学习在 global expert 与 routed local experts 之间分配容量 | 与 global+dense8 的完整四卡配对，不接受 static 或 rollout 退化 |
| router inhibition | 从 token router 输入中减去 pooled common signal，促进局部专家分工 | tiny benchmark 先筛选，再决定是否 full-scale |

最终保留标准不是单一 `valid rel-L2`：候选必须同时报告 low/high-band reconstruction、
高频泄漏、FNO/Transolver/CNextU latent rollout，以及解码后的 physical rollout。

`experiments/eval_operator_physical.py` 用对应 VAE 将完整 latent rollout 解码后评估物理场误差；`runs/` 下的 checkpoints、latent caches 和 CSV/JSON 是本次实验的原始产物，均未修改 `diff_pde`。

### Learnable MoE autoresearch 与 full-scale confirmation

在 H20 上用固定 tiny benchmark 做了 Git-backed autoresearch。第一阶段从
`robust_val_rel_l2=1.0091739` 开始，依次保留 global expert、两个 shared experts、
dense 4-expert routing、关闭 router noise、router 输入 LayerNorm 和初始
temperature `0.5`，最终为 `1.0007123`。8-expert dense、top-1/top-2/top-3、强化
balance、sigmoid gate、coordinate embedding、router z-loss、decoder-side 第二个
MoE 和非零 residual 初始化都没有优于保留配置；失败 trial 已由 autoresearch 回滚。
第二阶段从保留配置重新测量，目标 `0.95`，在 6 个候选假设后没有进一步改善，run
按 iteration limit 停止。tiny benchmark 的收益很小，因此最终选择不能只依赖该指标。

随后在完整 `data_128` 上用四卡训练 1 epoch，输入 `49x128x128`、latent
`(16,13,8,8)`，并用 30 epoch、history=4 的四卡 operator 协议验证：

| VAE variant | valid rel-L2 | predicted HF energy | FNO rollout | Transolver rollout | CNextU rollout |
| --- | ---: | ---: | ---: | ---: | ---: |
| noisy8 / top-2 | 0.05899 | 1.25e-5 | 0.25661 | 0.62069 | 0.22676 |
| fine16 / top-4 / 2-shared | 0.07206 | 1.37e-5 | 0.25665 | 0.60122 | 0.24019 |
| dense soft8 | 0.05977 | 1.28e-5 | 0.24641 | 0.58025 | 0.25389 |
| global + dense4 / 2-shared | 0.06475 | 1.42e-5 | 0.24790 | **0.57338** | **0.22609** |
| global + dense8 / 2-shared | **0.05768** | **1.23e-5** | 0.24922 | 0.57629 | 0.24017 |
| adaptive global/local + dense8 / 2-shared | 0.06796 | 1.27e-5 | 0.26151 | 0.59390 | 0.23949 |

解码到物理场后的 test rollout 为：

| VAE variant | FNO physical | Transolver physical | CNextU physical |
| --- | ---: | ---: | ---: |
| noisy8 / top-2 | 0.18457 | 0.45770 | 0.18136 |
| fine16 / top-4 / 2-shared | 0.19728 | 0.42003 | 0.20201 |
| dense soft8 | **0.18063** | **0.39480** | 0.20957 |
| global + dense4 / 2-shared | 0.18277 | 0.39516 | 0.18205 |
| global + dense8 / 2-shared | **0.17728** | **0.38789** | 0.19317 |
| adaptive global/local + dense8 / 2-shared | 0.19485 | 0.41497 | 0.19483 |

global 配置的 valid routing load 是严格均匀的 `0.25`，但 importance 为
`[0.385, 0.163, 0.344, 0.107]`，temperature 收敛到 `0.5007`。这说明 dense
MoE 的 hard load 均匀不等于专家功能相同；重要性仍出现内容条件的分工。完整结果的
结论是：global+dense8 进一步把 FNO/Transolver physical rollout 降到当前矩阵最低，
但它的 latent rollout 并没有同步变好，CNextU 也退化到 `0.19317`；noisy8 仍是
CNextU 的最低 physical 值。因此当前没有“对所有
operator 都占优”的单一 VAE，必须按目标下游模型选择配置。

### Operator 内部 hidden-state MoE

受 NESTOR 的 nested global/local operator 和 MoE Operator Transformer 的 layer-wise
routing 启发，新增了一个只作用于 operator hidden state 的实验。FNO 的每个 spectral
block、Transolver 的前三个 hidden token block、CNextU 的 encoder/neck/decoder stage
后面各接一个零初始化的 `LearnableTokenMoE`；输出端仍保留当前 top-2 global/shared
MoE。该模块不改 token 数量和空间位置，通过 `--operator-internal-moe` 显式启用。

在固定 `hf_ddp_global8` latent、seed=3407、30 epoch、四卡训练和 240 条 test
trajectory 的协议下，autoresearch 的 baseline 与内部 MoE 的 decoded physical
结果为：

| operator | output MoE only | output + internal MoE | change |
| --- | ---: | ---: | ---: |
| FNO | 0.174051 | **0.166324** | -4.44% |
| Transolver | **0.374761** | 0.378130 | +0.90% |
| CNextU | 0.181926 | **0.158129** | -13.08% |
| mean | 0.243579 | **0.234194** | -3.85% |

内部 hidden MoE 达到了 autoresearch 目标 `mean_physical_rel < 0.235`，但不能宣称
对所有 backbone 都改善：Transolver 的 latent rollout 变好而 decoded physical
误差略变差，说明 decoder/latent-to-physical 映射仍会重新放大某些误差。当前保留
internal MoE 作为 FNO/CNextU 候选，Transolver 仍使用 output-only 配置更稳妥。全层
MoE 的计算代价也明显更高，尤其 CNextU 的高通道 bottleneck stage；后续应测试只在
decoder 或低分辨率 stage 路由的容量折中。

stage 选择消融保持其他配置不变。FNO/Transolver 的 hidden block 仍全部加入 MoE，
只有 CNextU 改变注入范围：

| CNextU scope | latent test rollout | decoded physical rollout |
| --- | ---: | ---: |
| output MoE only | 0.229911 | 0.181926 |
| decoder only | 0.207422 | 0.170765 |
| neck + decoder | 0.201734 | 0.165751 |
| all encoder/neck/decoder | not separately retained | **0.158129** |

`neck + decoder` 已经优于 decoder-only，但仍不及全层注入；它是计算量和效果之间
的可用折中。上表中 all scope 的 CNextU latent rollout 未在早期日志中单独保存，
因此不填入未经直接记录的数值，physical 数值仍来自完整 test 评估。

### Operator-level routing 消融

在同一 `hf_ddp_global8` latent、seed=3407、30 epoch、8 experts、top-2、2 shared
experts、global context 和 all-stage internal MoE 配置下，只把路由从
`token_choice` 改为 `expert_choice`。后者让每个 expert 选择固定容量的 token，未被
选中的 token 仍经过 shared expert：

| routing | FNO physical | Transolver physical | CNextU physical | mean |
| --- | ---: | ---: | ---: | ---: |
| token-choice | 0.166324 | 0.378130 | 0.158129 | 0.234194 |
| expert-choice | 0.189372 | 0.365875 | 0.167789 | 0.241012 |
| dense MLP token-choice (`top-k=8`) | 0.164974 | 0.325435 | 0.175491 | 0.221967 |
| dense coordinate-kernel (`top-k=8`) | **0.145898** | **0.278773** | **0.152251** | **0.192307** |

expert-choice 相比 output-only baseline 的 mean `0.243579` 仍有小幅改善，但比
token-choice hidden MoE 的 `0.234194` 差 `2.91%`。它只改善了 Transolver，FNO 和
CNextU 均退化，说明固定 expert 容量不是当前规则网格 NS latent 的通用高频先验。
相反，dense MLP token-choice 让每个 token 对 8 个 expert 使用连续 softmax 权重，三模型
mean 从 `0.234194` 降到 `0.221967`，相对改善 `5.22%`；其中 Transolver 改善最大，
但 CNextU 从 `0.158129` 退化到 `0.175491`。这说明仅对逐 token MLP experts，CNextU
偏好 hard top-2；该判断不应外推到不同的专家函数。scorer 默认仍为 top-2，dense 配置
通过 `--operator-moe-top-k 8` 复现。该结果支持“可学习的连续专家组合”比固定 expert
容量更有潜力，但不支持任何一种路由对所有 backbone 都普适。

### Coordinate-kernel experts

此前 routed expert 只是逐 token MLP，因而 dense routing 的收益可能只来自更大的
逐点容量。受 CondConv 的条件化 kernel bank、Kernel Neural Operator 和 spatially
conditioned neural-operator MoE 启发，将 routed experts 改为坐标条件 kernel
operator，其 shared/global expert、router、8 experts、top-k=8、训练预算和 VAE latent
全部保持不变。对 token `i,j`，expert `e` 的消息权重为

\[
a_{ij}^{(e)}=\operatorname{softmax}_j\left(
q_i^{(e)\top}k_j^{(e)}/\sqrt d
-\frac{\|c_i-c_j\|^2}{2\ell_e^2}
+ b_e(c_i-c_j)\right),
\qquad
E_e(h_i)=\operatorname{MLP}_e\left(h_i+\sum_j a_{ij}^{(e)}V_e(h_j)\right).
\]

`\ell_e` 和相对坐标 bias 都是可学习的；该形式可以直接接受 arbitrary
`(value, coords, mask)` token，不预设规则网格、Fourier mode 或 slot。外部 residual
projection 仍为 zero-init，所以新 adapter 在初始化时严格等于原 operator。

在 `hf_ddp_global8` latent、seed=3407、30 epochs、四卡训练、240 条 test trajectory
decoded physical rollout 的完整对照中，kernel experts 对三种 backbone 均改善：FNO
从 `0.164974` 到 `0.145898`（-11.56%），Transolver 从 `0.325435` 到 `0.278773`
（-14.34%），CNextU 从 `0.175491` 到 `0.152251`（-13.24%）；mean 从 `0.221967`
到当前最佳 `0.192307`（-13.36%，相对 output-only baseline `0.243579` 为 -21.05%）。
对应 latent test rollout 是 FNO `0.193217`、Transolver `0.350143`、CNextU
`0.179408`，也全部优于 dense MLP。

路由诊断在 32 个 test batch 上显示，FNO/Transolver 的 internal experts 同时出现
非均匀 importance 和不同有效半径：FNO 约 `0.10-0.29`，Transolver 约 `0.11-0.28`。
这不是 uniform expert averaging。CNextU 的两个 `1x1` bottleneck stage 保持初始
`0.25` 半径，符合该空间分辨率下坐标距离无法提供额外信息的预期；其他 stage 仍产生
内容条件路由和可学习尺度。复现实验使用：

```bash
python experiments/score_operator_moe.py \
  --operator-moe-top-k 8 \
  --operator-moe-expert-kind kernel
```

这使 coordinate-kernel dense MoE 成为当前所有三种 operator 都占优的默认候选。它的
代价是 pairwise token kernel 使训练比 MLP expert 更慢。

### Adaptive coordinate-kernel experts

进一步把每个 expert 的固定尺度改成由当前有效 token 的全局状态条件化的样本级尺度：

\[
\ell_e(x)=\operatorname{clip}\left(
\ell_e\exp\left[0.5\tanh\left(r_e(\operatorname{mean}_{mask} h)\right)\right]
\right).
\]

`r_e` 以零初始化，因此初始模型严格退化为 fixed-kernel 配置；它不产生 slot，也不改变
token 数量、坐标或顺序。此处的条件化只让同一 expert 在不同流态下选择不同的连续感受野。

同一 latent、seed、30 epoch、四卡训练与 240 条 decoded physical rollout 下，结果为：

| expert | FNO physical | Transolver physical | CNextU physical | mean |
| --- | ---: | ---: | ---: | ---: |
| fixed coordinate-kernel | **0.145898** | 0.278773 | **0.152251** | 0.192307 |
| sample-adaptive coordinate-kernel | 0.151548 | **0.251153** | 0.159708 | **0.187469** |

adaptive 使 mean 从 `0.192307` 降到 `0.187469`（`-2.52%`，相对 output-only baseline
`0.243579` 为 `-23.03%`），但它的收益完全由 Transolver 的 `-9.19%` 驱动；FNO 和
CNextU 分别退化 `3.87%` 与 `4.90%`。对应 latent test rollout 分别为 FNO `0.197663`、
Transolver `0.325811`、CNextU `0.190996`，也呈现相同的 backbone-dependent 模式。
因此它是可保留的 Transolver 候选，而不是固定 coordinate-kernel 的全局替代。

32 个 test batch 的路由诊断证明 adaptive 参数确实被使用：expert 的 soft importance
明显不均匀，空间层的 learned mean radius 在不同 expert 间跨越约 `0.08-0.57`，且多个
stage 的样本内 radius standard deviation 为 `0.01-0.03`。CNextU 的两个 `1x1`
bottleneck stage 保持 `0.250244` 且标准差为 `0`，因为该处已没有可由坐标半径区分的空间
几何；这与实现预期一致。由于 dense top-8 会使 hard load 恒为均匀 `0.125`，这里以
soft importance 与 radius statistics，而非 hard load，作为是否发生 learnable expert
specialization 的证据。复现命令：

```bash
python experiments/score_operator_moe.py \
  --operator-moe-top-k 8 \
  --operator-moe-expert-kind adaptive_kernel
```

训练和评估代码对旧 checkpoint 做了兼容处理：`operator_moe` 与
`operator_internal_moe` 是两个独立结构标志，读取缺少新字段的旧 checkpoint 时只
重建 output adapter，不会凭 `operator_moe` 误加 hidden adapters。

### Adaptive global/local 失败假设

受 SPAR/U-HNO 中按空间位置学习 global Fourier branch 与 local multi-scale
branch 分配的启发，新增了一个不改变 token 数量和位置的 gate：
`g_i=sigmoid(G([h_i, pooled(h)]))`，并令 routed/global 两个分支做 token-wise
凸组合。gate 和 output residual 都从零初始化，旧 checkpoint 可以部分加载。该设计
在 tiny benchmark 上与 global+dense4 基线几乎持平（`1.0009075` 对 `1.0009013`），
但在完整四卡训练中 valid rel-L2 从 `0.05768` 退化到 `0.06830`，FNO、Transolver、
CNextU 的 latent rollout 和 decoded physical rollout 均退化。训练后 routing
importance 为 `[0.0647, 0.2410, 0.1459, 0.1442, 0.2947, 0.0563, 0.0305, 0.0227]`，
hard load 仍为均匀 `0.125`，global gate 平均值为 `0.4930`；这说明 gate/importance 的内容条件分化没有自动转化为
更好的高频表示，当前不应将 adaptive global/local 作为主线配置。

随后测试了 Neural Inhibition 风格的 router：用 `features-pooled(features)` 作为
局部分支的路由输入，同时保留 pooled context 供 router 使用。tiny benchmark 得到
`robust_val_rel_l2=1.0008997`，相同基线为 `1.0009013`，但 best validation 完全相同，
收益只有 `1.6e-6`，低于当前 tiny 实验的可辨识范围。因此该实现保留为下一轮候选，
暂不进行昂贵的四卡 full-scale 训练。

在 full-scale operator protocol 中真正接入 internal adapters 后，local-contrast
router 的结果如下：

| configuration | FNO physical | Transolver physical | CNextU physical | mean |
| --- | ---: | ---: | ---: | ---: |
| sample-adaptive coordinate-kernel | 0.151548 | 0.251153 | 0.159708 | **0.187469** |
| + router inhibition | 0.153828 | 0.258524 | 0.170696 | 0.194349 |

它相对 adaptive baseline 退化 `3.67%`，相对 fixed coordinate-kernel 也退化 `1.06%`。
这与 tiny benchmark 的微小正向差异不一致，说明在 NS rollout 中 pooled absolute state
包含重要的 regime/amplitude 条件；仅把路由输入改成 `features-pooled(features)` 会
损失这部分信息。该失败配置不再作为默认候选，复现开关为
`--operator-moe-router-inhibition`。

## 参考依据

- Deconvolution and Checkerboard Artifacts, Distill 2016: transposed convolution 容易产生棋盘伪影，resize-convolution 是常用替代。
- [Making Convolutional Networks Shift-Invariant Again, ICML 2019](https://proceedings.mlr.press/v97/zhang19a.html): 下采样前 anti-aliasing/BlurPool 能改善平移稳定性。
- On the Spectral Bias of Neural Networks, ICML 2019: 常规网络存在低频先学/高频慢学的频率偏置。
- [Sobolev Training for Neural Networks, NeurIPS 2017](https://arxiv.org/abs/1706.04859): 同时监督函数值和导数可编码更强目标信息。
- [Fourier Neural Operator for Parametric PDEs, ICLR 2021](https://arxiv.org/abs/2010.08895): FNO 直接在 Fourier 模态上学习 PDE solution operator。
- Physics-Informed Neural Operator, 2021: 将数据 loss 和 PDE loss 结合，适合作为后续突破“重建好但残差高”的路径。
- Transolver, 2024: physics-attention 通过物理 slice/token 捕捉复杂几何或场相关性。
- [Wavelet Neural Operator](https://arxiv.org/abs/2205.02191): 用空间局部化的小波尺度替代纯全局 Fourier 模态。
- [Geo-FNO](https://arxiv.org/abs/2207.05209): 用 learned deformation 把一般几何映射到适合 FFT 的潜在规则坐标。
- [M²M](https://arxiv.org/abs/2410.11617): 用多尺度、多专家和动态 gate 处理 PDE 中不同区域的振荡尺度。
- [Geometry-Informed Neural Operator](https://proceedings.neurips.cc/paper_files/paper/2023/file/70518ea42831f02afc3a2828993935ad-Paper-Conference.pdf): 结合几何编码和 Fourier/局部算子处理复杂几何。
- [From Sparse to Soft Mixture of Experts](https://openreview.net/pdf?id=jxpsAj7ltE): 用可学习 soft token mixing 避免硬 token dispatch 和 token dropping。
- [CondConv](https://arxiv.org/abs/1904.04971): 通过输入条件化权重混合实现 dynamic expert kernels。
- [Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538): noisy top-k gating 与 learned sparse routing 的原始形式。
- [Switch Transformer](https://arxiv.org/abs/2101.03961): top-1 routing、router 稳定化和负载平衡的规模化实践。
- [Expert Choice Routing](https://arxiv.org/abs/2202.09368): 由 expert 选择固定容量 token，避免 token-choice 的容量竞争。
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066): fine-grained routed experts 加 shared experts，提升组合式专家专门化。
- [NESTOR](https://arxiv.org/abs/2602.22059): 以 shared/global/local 层级构造 nested MoE neural operator。
- [Mixture of Neural Operator Experts](https://arxiv.org/abs/2502.04562): 用空间条件 MoE
  处理边界条件与模型选择，是 coordinate-kernel expert 的直接结构动机。
- [U-HNO / SPAR](https://arxiv.org/abs/2605.12965): 学习每个位置在 global Fourier 与 local multi-scale Gaussian 分支之间的分配，是 adaptive global/local 实验的直接动机。
- [Neural Inhibition](https://arxiv.org/abs/2507.03221): 通过抑制 router 中跨模式共享的公共信号促进 expert specialization，可作为下一轮路由正则化候选。
- [ABLE](https://arxiv.org/abs/2605.10451): 学习数据相关的 Parseval frame，说明固定 Fourier basis 的限制可以从“学习基底”而不只是“增加 modes”来缓解。
- [SFO](https://arxiv.org/abs/2601.17090): 用 Universal Spectral Basis 做谱滤波，强调适配 kernel 结构的基底比盲目扩大 Fourier 截断更重要。
- [GITO](https://arxiv.org/abs/2506.13906): 将图局部关系、Transformer 长程依赖和任意 query location 结合，适合作为 irregular branch 的 operator 对照。
- [FreqMoE](https://arxiv.org/abs/2505.06858): 以逐步扩展的稀疏频率专家处理 PDE 的多尺度误差，说明频率相关容量分配可以独立于固定 dense FNO。
- [Kernel Neural Operators](https://arxiv.org/abs/2407.00809): 用可学习紧支撑 kernel 和 quadrature 支持不规则点集，是 arbitrary-format high-frequency branch 的直接候选。

### 下一轮可执行的多模型实验

在已有 FFT/local cross-band 与 token-MoE 之外，优先级如下：

1. 用 `learnable token MoE` 比较 token-choice/expert-choice、不同初始化、top-1/top-2、专家数和 routing balance，
   同时记录 expert importance/load，检查高频区域是否真的被单独路由，而不是所有 token
   塌缩到同一 expert。
2. 加入 nested global/local 路径：global context 只提供低频和长程条件，local token
   experts 保留高频残差；这对应 NESTOR/M²M 的结构，但保留 zero-init latent contract。
3. 在规则 `data_128` 上做 adaptive-basis ablation：让低频 expert 学习少量 radial
   filter/basis 参数，而不是固定 cutoff；目标是区分“FFT 截断不足”和“固定 basis 不合适”。
4. 对不规则采样只比较支持坐标的 GITO/GNO/GNOT/GINO 类分支，报告 `n_obs`、边界偏置、
   query resolution 和缺失点 mask 的鲁棒性；不把 irregular 点强行 reshape 成 FFT 网格。
5. 对 internal MoE 做 stage 选择消融：FNO/Transolver 保留 hidden blocks，CNextU 只在
   decoder 或低分辨率 bottleneck 路由，单独报告参数量、吞吐和 decoded physical error。

最终选择仍然以三层证据为准：VAE 的 low/high reconstruction、latent FNO/Transolver/
CNextU rollout，以及解码后的 physical rollout；单一 static rel-L2 改善不能作为保留依据。
