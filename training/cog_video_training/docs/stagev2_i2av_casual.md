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
| SA denoise d6 continuous | `v5_depth_rgb_2524_stage2_sa_denoise_d6cont` | 将 SA 改成真正 denoising 目标 | 连续 action 通道 | transformer LoRA + aux | token-space noise prediction |

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

### 当前状态

该版本是最新一版，目标是验证真正 denoising SA token 是否能解决 d0-d5 不如 zero baseline 的问题。训练脚本设置为：

```text
partition=preempt
time=08:00:00
TRAIN_STEPS=6000
CHECKPOINTING_STEPS=500
```

需要重点观察：

```text
d0-d5 RMSE / zero 是否低于 1.0
d0-d5 corr 是否明显高于 0
pred std / gt std 是否不再过低
d6 continuous F1 是否保持稳定
state RMSE / zero 是否继续优于 zero
```

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

1. `raw_action_lora` 不建议继续投入。继续训练后 d0-d5 和 d6 都没有稳定改善。
2. `lingbot_bce` 对 d0-d5 有一定帮助，但 d6 BCE 出现全 0 坍缩，不适合作为最终路线。
3. `lingbot_d6cont` 是 clean-regression 系列里最合理的版本，验证了 d6 应该作为连续 action 通道建模。
4. 下一步重点应放在 `stage2_sa_denoise_d6cont`，因为它把 SA 从 decoded clean regression 改成和 diffusion/DiT 更一致的 denoising 目标，并配套修改了推理流程。

