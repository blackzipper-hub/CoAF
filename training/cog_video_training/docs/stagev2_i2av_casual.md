# Stage2 I2AV 版本与训练逻辑说明

本文档说明当前 `i2av_pt v5_depth_rgb_2524` 的 Stage2 各个实验版本：每一版是在遇到什么问题后提出的、训练逻辑如何变化、推理逻辑如何配套变化，以及当前实验结论。

## Stage2 的共同目标

Stage2 的目标不是继续训练视频生成质量，而是在给定 clean/GT video latent 的条件下学习 State/Action 轨迹。当前训练代码中，`train_stage=stage2` 时：

```text
loss_video = 0
loss = L_sa
```

也就是说，Stage2 里 video loss 不参与参数更新。模型看到的是干净视频 latent 与加噪后的 SA token，然后优化 SA 分支相关目标。

Stage2 推理也应与训练一致：使用 GT video 编码得到 clean video latent，不再让视频 latent 从噪声开始生成；推理重点是预测 state/action。对应推理输出按 `eval_dataset/{validation,train}/{gt,pred}/episode_xxxx` 保存，用于对 train/test 分开评估。

## 版本总览

| 版本 | checkpoint family | 主要目的 | d6 处理 | 训练对象 | SA 目标 |
|---|---|---|---|---|---|
| 原始 Stage2 | `v5_depth_rgb_2524_stage2_v1` | 验证原始 state/action 设计 | state delta 的第 6 维 | aux 为主 | decoded state + state delta regression |
| raw action aux-only | `v5_depth_rgb_2524_stage2_raw_action` / `raw_action_1gpu_1k` | 修正 action 监督语义 | BCE/二值 | aux only | decoded clean raw action regression |
| raw action LoRA | `v5_depth_rgb_2524_stage2_raw_action_lora` | 解决 aux head 表达力不足 | BCE/二值 | transformer LoRA + aux | decoded clean raw action regression |
| LingBot-style BCE | `v5_depth_rgb_2524_stage2_raw_action_lingbot` | 引入 mask/per-dim loss，降低维度平均稀释 | BCE/二值 | transformer LoRA + aux | decoded clean raw action regression |
| LingBot-style d6 continuous | `v5_depth_rgb_2524_stage2_raw_action_lingbot_d6cont` | 避免 d6 二分类偏置/全 0 坍缩 | 连续 action 通道 | transformer LoRA + aux | decoded clean raw action regression |
| SA denoise d6 continuous | `v5_depth_rgb_2524_stage2_sa_denoise_d6cont` | 将 SA 改成真正 denoising 目标 | 连续 action 通道 | transformer LoRA + aux | token-space noise prediction（初版有 scheduler/decoder 不一致问题） |
| SA denoise + quantile norm | `v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt` | quantile 归一化 + v-pred clean SA token denoise | 连续 action 通道 | transformer LoRA + aux | token-space clean SA prediction |
| **SA denoise qnt fix1（当前主线）** | `v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt_fix1` | 在 qnt 基础上补 decoded action/state 辅助监督 + 推理 SA CFG 解耦 | 连续 action 通道 | transformer LoRA + aux | v-pred clean SA + decoded action/state loss |

## 1. 原始 Stage2：`stage2_v1`

相关脚本：

```text
jobs/train/i2av_pt/train_i2av_pt_v5_depth_rgb_2524_stage2_5k_60k.sbatch
```

### 想解决的问题

原始设计希望 Stage2 在 clean video 条件下学习 state/action token。它延续了早期设计：action 不是读取数据集里的 `action.npy`，而是由 state 序列差分得到：

```text
action_gt = state[t + 1] - state[t]
```

### 训练逻辑

`prepare_gt_chunked()` 将 state 对齐到 v5 chunk，然后构造：

```text
state_gt = normalized(state)
action_gt = normalized(state_delta)
```

loss 是 decoded token 后的回归：

```text
L_sa = lambda_s * L_state + lambda_a * L_action + lambda_c * L_consistency
```

### 暴露的问题

这个版本最大的问题是 **action 监督语义不对**。模型学到的是 state delta，不是真实机器人控制量 `action.npy`。后续评估时拿预测 action 和 raw action 对比，就会出现语义错位。

进一步分析发现，`state delta` 和真实 `action.npy` 在 d0-d6 上差异明显，尤其 d6 gripper 根本不是一个可靠的 state delta 通道。因此用 state delta 代替 action 不合理。

## 2. Raw Action Aux-only：`stage2_raw_action`

相关脚本：

```text
jobs/train/i2av_pt/train_i2av_pt_v5_depth_rgb_2524_stage2_raw_action_1gpu_1k.sbatch
jobs/train/i2av_pt/train_i2av_pt_v5_depth_rgb_2524_stage2_raw_action_3k.sbatch
```

### 想解决的问题

为了解决原始 Stage2 的 action 语义错位，训练目标改为直接读取 batch 中的 `action.npy`，即 raw action 作为主监督。state delta 不再作为 action label，只保留为 consistency 辅助项。

### 训练逻辑

新增 `action_norm_stats.pt`，对 raw action 做归一化：

```text
d0-d5: normalized continuous action
d6: BCE gripper label
```

loss 变为：

```text
L_sa = lambda_s * L_state
     + lambda_a * L_action
     + lambda_g * L_gripper
     + lambda_c * L_consistency
```

其中：

```text
L_action: d0-d5 SmoothL1
L_gripper: d6 BCEWithLogits
L_consistency: pred state implied delta vs GT state delta
```

### 暴露的问题

aux-only 版本只训练 I2AV 的辅助模块，不训练 transformer LoRA。实验显示 d0-d5 仍然严重接近低动态输出，无法超过 zero baseline；d6 也容易偏向单一类别。

结论是：只修正 label 语义还不够，aux head 表达能力不足。

## 3. Raw Action LoRA：`stage2_raw_action_lora`

相关脚本：

```text
jobs/train/i2av_pt/train_i2av_pt_v5_depth_rgb_2524_stage2_raw_action_lora_3k.sbatch
jobs/train/i2av_pt/train_i2av_pt_v5_depth_rgb_2524_stage2_raw_action_lora_continue_6k_normal.sbatch
```

### 想解决的问题

aux-only 不足后，打开 transformer LoRA，让主干也参与 action/state 学习：

```text
STAGE2_TRAIN_TRANSFORMER_LORA=1
```

### 训练逻辑

仍然使用 raw action regression，但训练对象变为：

```text
transformer LoRA + i2av aux modules
```

该版本还提高了 action/gripper 权重：

```text
LAMBDA_A=2.0
LAMBDA_G=5.0
LAMBDA_C=0.05
```

### 暴露的问题

LoRA 参与后，d6 相比 aux-only 有明显反应，但 d0-d5 并没有真正学到动态轨迹。继续训练到更高 step 后，d0-d5 指标反而变差：

```text
checkpoint-3000 validation d0-d5 RMSE / zero: 约 1.81x
checkpoint-5500 validation d0-d5 RMSE / zero: 约 2.32x
```

这说明问题不是单纯“步数不够”，而是 clean action regression 在小幅连续 action 上仍容易学到偏置或错相位输出。

## 4. LingBot-style BCE：`stage2_raw_action_lingbot`

相关脚本：

```text
jobs/train/i2av_pt/train_i2av_pt_v5_depth_rgb_2524_stage2_raw_action_lingbot_3k_normal.sbatch
jobs/train/i2av_pt/train_i2av_pt_v5_depth_rgb_2524_stage2_raw_action_lingbot_continue_6k_normal.sbatch
```

### 想解决的问题

前面版本的 loss 对 action 维度是简单平均，容易被低幅值维度、无效维度或 gripper 偏置影响。该版本借鉴 LingBot 的思想，引入：

```text
valid_action_mask
per-dim normalized mean
raw action as main supervisor
state delta only as consistency
```

### 训练逻辑

核心仍是 decoded clean action regression，但 action loss 不再简单全元素平均：

```text
cont_loss = SmoothL1(pred_action[..., :6], action_gt[..., :6])
L_action = masked per-dim mean(cont_loss)
L_gripper = weighted BCEWithLogits(d6)
L_consistency = masked state-delta consistency
```

### 暴露的问题

d0-d5 比 raw_action_lora 更接近 zero baseline，但 d6 BCE 出现严重偏置，预测几乎全 0：

```text
checkpoint-5500 validation d6 F1 = 0
pred positive rate = 0
```

说明 BCE gripper 在当前分布和权重下不稳定，会形成类别坍缩。

## 5. LingBot-style d6 Continuous：`stage2_raw_action_lingbot_d6cont`

相关脚本：

```text
jobs/train/i2av_pt/train_i2av_pt_v5_depth_rgb_2524_stage2_raw_action_lingbot_d6cont_3k_normal.sbatch
```

### 想解决的问题

LingBot 原版把 gripper 也作为连续 action 通道建模，而不是单独做 BCE。为了避免 d6 的二分类偏置，将 d6 放回 action token，像 d0-d5 一样归一化并回归。

### 训练逻辑

开启：

```text
GRIPPER_CONTINUOUS_ACTION=1
```

此时：

```text
d0-d6 全部作为连续 action 通道
L_gripper = 0
d6 loss 并入 L_action
```

### 当前结论

这是 clean-regression 系列里综合最好的版本。它解决了 BCE d6 全 0 的问题，d6 预测正例率能接近 GT。阶段性评估中：

```text
checkpoint-2500 validation d6 F1: 约 0.64
state RMSE / zero: 约 0.70x
```

但是 d0-d5 仍然没有超过 zero baseline：

```text
checkpoint-2500 validation d0-d5 RMSE / zero: 约 1.19x
```

因此，d6 continuous 是正确方向，但直接 decoded clean action regression 仍然不足以学好小幅连续 action 轨迹。

## 6. SA Denoise d6 Continuous：`stage2_sa_denoise_d6cont`

相关脚本：

```text
jobs/train/i2av_pt/train_i2av_pt_v5_depth_rgb_2524_stage2_sa_denoise_d6cont_6k_preempt.sbatch
jobs/infer/i2av_pt/infer_i2av_pt_v5_depth_rgb_2524_stage2_sa_denoise_d6cont.sbatch
```

### 想解决的问题

前面的版本虽然输入端有 noisy SA token，但 loss 仍然是 decoded clean state/action regression：

```text
noisy_sa -> transformer -> sa_pred -> decode -> clean action/state loss
```

这和 LingBot 的 action flow/denoising 目标还有差距。对于 d0-d5 这种幅度很小的连续动作，clean regression 很容易学习 conditional mean，导致输出接近低动态均值，仍然输给 zero baseline。

### 新训练逻辑

新增开关：

```text
SA_DENOISE_LOSS=1
```

训练时先构造 clean SA token：

```text
clean_sa = sa_tokenizer.encode(state_gt, action_gt)
noise_sa = randn_like(clean_sa)
noisy_sa = scheduler.add_noise(clean_sa, noise_sa, timesteps)
```

然后模型直接预测加入 SA token 的噪声：

```text
sa_pred = transformer(... noisy_sa ...)
L_sa = MSE(sa_pred, noise_sa)
```

为了观察 state/action token 的学习情况，日志中分别记录：

```text
L_state: state token denoise MSE
L_action: action token denoise MSE
L_sa_denoise: weighted denoise loss
```

这个版本仍然使用：

```text
GRIPPER_CONTINUOUS_ACTION=1
STAGE2_TRAIN_TRANSFORMER_LORA=1
```

### 新推理逻辑

对应推理也必须从随机 SA token 开始做多步 denoising，而不是把 `sa_pred` 直接当 clean token：

```text
sa_tokens = random noise
for timestep in timesteps:
    sa_noise_pred = transformer(... sa_tokens ...)
    sa_tokens = scheduler.step(sa_noise_pred, timestep, sa_tokens)
pred_state, pred_action = sa_tokenizer.decode(sa_tokens)
```

这使训练目标和推理流程一致：训练预测 SA noise，推理用 scheduler 逐步去噪。

### 当前结论

初版 `sa_denoise_d6cont` 在训练/推理目标对齐上存在问题：训练侧曾按 noise prediction 实现，但 CogVideoX scheduler 实际使用 v-prediction；推理侧也未完全按同一 denoise 语义采样。对比 `lingbot_d6cont`（checkpoint-2000）：

| 指标 (validation, 14 ep) | lingbot_d6cont | denoise_d6cont |
|---|---|---|
| d0-d5 RMSE / zero | 1.15x | 1.21x |
| d0-d5 corr | ≈ 0 | ≈ -0.05 |
| d6 F1@0.5 | 0.70 | 0.78（但 pred_pos=1.0，偏置严重） |
| pred_std / gt_std (d0-d5) | 0.53 | 0.18（动态被压得更低） |

该路线说明：**仅把 loss 改成 token denoise 还不够，必须同时修正 v-pred 目标、decoded 监督与推理采样一致性**。后续 `qnt` / `fix1` 在此基础上继续迭代。

## 7. SA Denoise + Quantile Norm：`stage2_sa_denoise_d6cont_qnt`

相关脚本：

```text
jobs/train/i2av_pt/train_i2av_pt_v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt_6k_normal.sbatch
jobs/infer/i2av_pt/infer_i2av_pt_v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt.sbatch
scripts/build_action_quantile_norm_stats.py
```

### 想解决的问题

1. 原始 `action_norm_stats.pt`（mean/std）对小幅连续 action 的尺度不敏感，d0-d5 容易被低动态均值淹没。
2. 初版 SA denoise 的 scheduler/decoder 目标不一致，需要统一到 v-pred clean SA token。

### 训练逻辑

在 `sa_denoise_d6cont` 基础上：

```text
ACTION_NORM_STATS = action_quantile_norm_stats.pt
d0-d6: (x - q01) / (q99 - q01) * 2 - 1, clip 到 [-1.5, 1.5]
SA_DENOISE_LOSS = 1
GRIPPER_CONTINUOUS_ACTION = 1
STAGE2_TRAIN_TRANSFORMER_LORA = 1
```

loss 核心为 **v-pred clean SA token MSE**（state/action token 分开加权）：

```text
sa_output = scheduler.get_velocity(model_output, noisy_sa, timesteps)  # clean SA estimate
L_sa_denoise = lambda_s * L_state_token + lambda_a * L_action_token
```

默认权重：`LAMBDA_S=1.0`, `LAMBDA_A=2.0`。

### 全量训练 sweep 结论（validation, 14 test + 8 train）

| checkpoint | d0-d5 RMSE/zero | d0-d5 corr | d6 F1@0.5 | 备注 |
|---|---|---|---|---|
| 2k | 1.42x | -0.056 | 0.61 | 早期仍输给 zero |
| 3k | 1.21x | -0.039 | 0.61 | 略有改善 |
| 6k | 1.12x | 0.003 | 0.75 | d6 最好阶段之一 |
| 8.5k* | 1.12x | 0.045 | 0.28 | d6 开始坍缩 |
| 10k | 1.33x | 0.055 | 0.33 | d0-d5 回退 |
| 15k | 1.09x | 0.027 | 0.17 | d6 pred_pos 仅 8% |

结论：

1. quantile norm + v-pred denoise **比初版 denoise 和 lingbot 系列更稳定**，长训后 d0-d5 RMSE/zero 可压到约 1.09x，但仍**未稳定低于 1.0**。
2. d0-d5 pooled corr 始终接近 0，说明全量数据上**方向/相位对齐仍失败**。
3. d6 在中期（6k）表现最好，继续训练后出现**正例率坍缩**（15k 时几乎不预测 open）。

## 8. SA Denoise QNT Fix1：`stage2_sa_denoise_d6cont_qnt_fix1`（当前主线）

相关脚本：

```text
jobs/train/i2av_pt/train_i2av_pt_v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt_fix1_6k_normal.sbatch
jobs/infer/i2av_pt/infer_i2av_pt_v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt_fix1.sbatch
jobs/train/i2av_pt/train_i2av_pt_v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt_fix1_one_sample_2k.sbatch
jobs/infer/i2av_pt/infer_i2av_pt_v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt_fix1_one_sample_2k.sbatch
```

### 想解决的问题

`qnt` 虽然统一了 v-pred 目标，但仅有 token-space denoise 时，decoder 到 7D action 的梯度仍然偏弱，尤其 rotation 维几乎学不到；同时推理若沿用 video CFG scale，SA 分支容易被过引导。

### 相对 `qnt` 的改动

训练侧新增 decoded 辅助监督：

```text
L_sa = L_sa_denoise
     + LAMBDA_DECODED_STATE  * L_decoded_state    # 默认 0.1
     + LAMBDA_DECODED_ACTION * L_decoded_action   # 默认 1.0
```

其中 `L_decoded_action` 直接对 `sa_tokenizer.decode(sa_output)` 与 GT raw action（quantile norm 后）做 SmoothL1。

推理侧将 SA CFG 与 video CFG 解耦：

```text
SA_GUIDANCE_SCALE = 1   # 默认不再用 video guidance_scale=6 过推 SA
```

其余保持：`SA_DENOISE_LOSS=1`, `GRIPPER_CONTINUOUS_ACTION=1`, quantile norm, transformer LoRA。

### 全量训练结果（checkpoint-14500，validation 14 ep / train 8 ep）

| 指标 | validation | train |
|---|---|---|
| d0-d5 RMSE / zero | **1.15x** | 1.28x |
| d0-d5 corr (pooled) | **0.043** | 0.067 |
| d6 corr | **0.579** | 0.678 |
| d6 F1@0.5 | **0.839** | 0.801 |
| d6 pred_pos / gt_pos | 0.77 / 0.64 | 0.84 / 0.57 |
| 7D corr (pooled) | **0.802** | 0.858 |
| state RMSE / zero | 0.85x | 1.23x |

逐维 corr（validation）：

| dim | d0 (x) | d1 (y) | d2 (z) | d3 (rx) | d4 (ry) | d5 (rz) | d6 (gripper) |
|---|---|---|---|---|---|---|---|
| corr | 0.14 | 0.19 | **0.35** | 0.07 | 0.02 | 0.01 | **0.58** |

逐维 corr（train）：

| dim | d0 | d1 | d2 | d3 | d4 | d5 | d6 |
|---|---|---|---|---|---|---|---|
| corr | 0.48 | **0.65** | 0.43 | 0.05 | 0.08 | 0.16 | 0.68 |

fix1 阶段性结论：

1. **d6 明显优于 qnt 长训版本**：F1 从 15k 的 0.17 恢复到 0.84，说明 decoded action 监督和 SA CFG 解耦有效。
2. **xyz 有一定信号，rotation 几乎为零**：d0-d2 在 train 上 corr 可达 0.43-0.65，但 d3-d5 在 validation 上仅 0.01-0.07。
3. **7D pooled corr 高但具有误导性**：主要由 d6 大尺度通道拉高；d0-d5 pooled corr 仍接近 0。
4. **仍未稳定击败 zero baseline**：d0-d5 RMSE/zero ≈ 1.15x，说明全量训练下小幅平移动作仍未对齐。

### One-sample 过拟合 sanity check（fix1, 2000 step, 256×256）

在同一 `episode_000000` 上单独过拟合 2000 step，可验证 pipeline 本身具备学习能力：

| 指标 | validation (1 ep) |
|---|---|
| d0-d5 corr | **0.733** |
| d0-d5 RMSE / zero | **0.73x**（击败 zero） |
| d3-d5 corr | **0.69 / 0.72 / 0.69** |
| d6 corr | **0.808** |
| d6 F1@0.5 | **0.842** |
| 7D corr | **0.842** |

这说明：**模型与数据管线可以过拟合单条样本，包括 rotation**；全量训练 rotation 差，更可能是 loss 设计、多样本信号稀释、或 rotation 监督权重不足，而不是架构完全学不动。

## 为什么 zero baseline 很强

当前 d0-d5 的 GT action 幅度很小，validation 上 zero baseline RMSE 约为：

```text
d0-d5 zero RMSE ~= 0.024
```

这意味着模型只要预测的方向、相位或幅值稍微错一点，就可能比全 0 更差。输给 zero baseline 不一定代表完全没有输出，而是说明预测动态没有对齐 GT，或者回归目标使模型学习到了低动态均值。

因此后续判断不能只看 loss 降不降，还要看：

```text
RMSE / zero
correlation
pred_std / gt_std
per-dim RMSE
d6 positive rate / F1
```

## 当前推荐结论

1. `raw_action_lora` / `lingbot_bce` 不建议继续投入：前者长训后 d0-d5 恶化，后者 d6 BCE 全 0 坍缩。
2. `lingbot_d6cont` 验证了 **d6 应作为连续 action 通道**，是 clean-regression 系列最合理基线。
3. 初版 `sa_denoise_d6cont` 已过时：scheduler/decoder 不一致，且 pred 动态过低。
4. **`sa_denoise_d6cont_qnt_fix1` 是当前主线**：d6 和 xyz 有可见改善，但全量数据上 d0-d5 仍未稳定击败 zero，rotation 几乎学不到。
5. one-sample 过拟合实验证明 **pipeline 可学 rotation**；全量训练问题更可能在 loss/监督设计，而非模型容量。

## 下一步可能方向

按优先级排序：

### P0：先把 fix1 训满并稳定评估

```text
resume fix1 到 15000 step
固定 eval 协议：14 test + 8 train，报告 d0-d5 / per-dim corr / d6 F1 / pred_std
```

当前最新 checkpoint 为 `checkpoint-14500`（训练 job 在 step 14979 附近 TIMEOUT）。补完最后 500 step 后再做一次完整推理对比。

### P1：针对 rotation 维加强监督

观察：全量 fix1 上 d3-d5 corr ≈ 0，但 one-sample 可达 0.69-0.74。

可尝试：

```text
per-dim loss weight: 提高 d3-d5 权重（如 3-5x）
rotation-only auxiliary head 或单独 rotation decoder loss
decoded action loss 从 SmoothL1 改成 per-dim weighted SmoothL1
```

### P2：Action-space 直接输出（参考 LingBot-VA）

当前路径：`transformer → SA chunk tokens [B,7,8,3072] → MLP → 7D action`。

LingBot-VA 直接在 action 空间做 flow/denoise，不经 3072-dim SA token decoder。计划中的改法是：

```text
noisy action [B,T,7] → chunk encoder → action tokens
transformer → action tokens → velocity head → [B,T,7]
loss: action velocity / x0 regression（绕过 SA token decode 瓶颈）
```

这有望让 rotation 监督更直接，减少 token decode 带来的信息损失。

### P3：归一化与数据分布

```text
继续用 quantile norm（已优于 mean/std）
检查 rotation 维在 q01/q99 压缩后是否过度平坦
考虑 rotation 单独归一化或保留原始角度尺度
```

### P4：训练策略

```text
小样本过拟合 → 8-sample overfit → 再扩全量（验证 loss 改动是否有效）
课程学习：先训 xyz+d6，再放开 rotation
对比 SA_GUIDANCE_SCALE / LAMBDA_DECODED_ACTION 网格
```

### 暂不建议的方向

- 单纯提高输入分辨率：已验证上采样不能改善 rotation，且会增加算力开销。
- 回到 d6 BCE：已知会类别坍缩。
- 仅增加训练步数而不改 loss：qnt 15k 已出现 d6 退化，说明长训不是万能药。

## 指标文件索引

```text
outputs/infer/i2av_pt/lingbot_vs_denoise_compare_metrics.json
outputs/infer/i2av_pt/quantile_sweep_compare_metrics.json
outputs/infer/i2av_pt/fix1_14500_compare_metrics.json
outputs/infer/i2av_pt/one_sample_overfit_2k_metrics.json
```

