# Casual I2AV 模型架构说明

本文档描述 **Casual CoAF** 项目中 **Causal Image-to-Action-Video (I2AV)** 模型的完整架构：基于 CogVideoX-5B-I2V 的 DiT，在因果自注意力框架下联合去噪 **视觉 latent**、**7DoF 绝对状态** 与 **7DoF 动作增量**。

相关实现代码：

| 模块 | 路径 |
|------|------|
| 训练入口 | `finetrainers/examples/_legacy/training/cogvideox/cogvideox_image_to_video_lora_i2av.py` |
| State/Action 编解码 | `finetrainers/finetrainers/patches/models/cogvideox/state_action.py` |
| Token 交错 | `finetrainers/finetrainers/patches/models/cogvideox/i2av_sequence.py` |
| 自定义 Forward | `finetrainers/finetrainers/patches/models/cogvideox/i2av_forward.py` |
| 因果注意力 | `finetrainers/finetrainers/patches/models/cogvideox/causal_attention.py` |
| RoPE 扩展 | `finetrainers/examples/_legacy/training/cogvideox/utils.py` |
| 视频拼接 | `coaf_dataset/scripts/compose_all.py` |

---

## 1. 设计目标

I2AV 在标准 **I2V（Image-to-Video）** 之上扩展为 **Image-to-Action-Video**：

1. **输入**：首帧 RGB 图像 + 文本指令 + 初始关节角 \(s_0\)
2. **输出**：联合生成的 **多模态视频 latent** + **13 步状态/动作轨迹**
3. **核心思想**：视觉、状态、动作共享同一个 DiT 与同一个扩散时间步，通过 **因果 self-attention** 建模 MDP 时序——先观察画面，再确认状态并决策动作，再产生下一帧画面。

与纯 I2V 相比，I2AV 不额外堆叠独立 action head，而是把 state/action 当作 **与视觉 patch 同级的 token**，插入每帧 latent 之后，由 DiT 原生 attention 完成跨模态推理。

---

## 2. 输入视频：Pose 段与 RGB 段的帧数与排列

训练数据由 `compose_all.py` 将 **辅助模态视频** 与 **RGB 真机视频** 沿时间维拼接为一条 mp4。

### 2.1 各数据集版本的帧布局

| 版本 | 模态段 | Pose/Depth/Flow 段 | RGB 段 | 总像素帧数 | 训练 `MAX_NUM_FRAMES` |
|------|--------|-------------------|--------|-----------|----------------------|
| v1_pose_rgb | pose | **24** | **25** | **49** | 49 |
| v2_flow_rgb | flow | 24 | 25 | 49 | 49 |
| v4_depth_rgb | depth | 24 | 25 | 49 | 49 |
| v5_pose_depth_rgb | pose + depth | 24 + 24 | 25 | **73** | 73 |
| v3_pose_flow_rgb | pose + flow | 24 + 24 | 25 | 73 | — |

默认参数：`--cond-frames 24`（每个辅助模态采样 24 帧），`--rgb-frames 25`（RGB 固定 25 帧）。

### 2.2 时间轴排列（以 v1_pose_rgb 为例）

```
像素帧索引:  0    1    2   ...  23  |  24   25  ...  48
            ├──── pose 段 (24帧) ────┤  ├── RGB 段 (25帧) ──┤
内容:        机器人姿态 silhouette     |  真实 RGB 相机画面
```

拼接逻辑（`compose_all.py`）：

```python
combined = np.concatenate(modality_frames_list + [rgb_frames], axis=0)
# 先所有辅助模态，再 RGB；RGB 始终在最后
```

**条件图像**：取 RGB 段第 0 帧（`rgb_frames[0]`）作为 I2V 条件图，保存为 `condition_images/episode_XXXXXX.png`。

### 2.3 像素帧 → DiT Latent 帧

CogVideoX VAE 时间压缩比为 4：

\[
F_{\text{latent}} = \frac{F_{\text{pixel}} - 1}{4} + 1
\]

| 像素帧数 | Latent 帧数 |
|---------|------------|
| 49 (v1/v2/v4) | **13** |
| 73 (v3/v5) | **19** |

以 **49 帧 / 13 latent** 为主配置（当前大部分 I2AV 训练 job）。

### 2.4 状态序列与视频帧的对齐

- 原始 `state.npy`：**24 帧 × 7DoF** 绝对关节角（与 pose 段帧数一致，来自 Bridge 轨迹）
- 训练时在 `prepare_gt()` 中线性插值下采样到 **13 帧**，与 **13 个 latent 时间步** 对齐
- Action GT 不由原始 `action.npy` 直接监督，而是由相邻 state 差分导出：\(a_t = s_{t+1} - s_t\)（归一化后）

```
24-frame state  ──linspace──►  13-frame state_gt
                              └──► action_gt = diff(state), 末帧 pad 0
```

---

## 3. DiT 骨干：CogVideoX-5B-I2V

### 3.1 整体结构

```
┌─────────────────────────────────────────────────────────────┐
│                    CogVideoXTransformer3D                    │
├─────────────────────────────────────────────────────────────┤
│  patch_embed(text + video latents)  →  token embeddings      │
│  time_embedding(timestep)           →  AdaNorm 条件          │
│  transformer_blocks × N (默认 42 层)                          │
│    ├─ attn1: Self-Attention (+ RoPE, 因果 mask)            │
│    ├─ attn2: Cross-Attention (video → text/S0 条件)          │
│    └─ FFN                                                      │
│  norm_out + proj_out  →  预测 video velocity                   │
└─────────────────────────────────────────────────────────────┘
         ▲                              │
         │ I2AV 扩展                      ▼
    S0Encoder / StateActionTokenizer   SA token 输出（不经 proj_out）
```

- **基座**：`THUDM/CogVideoX-5b-I2V`
- **微调方式**：LoRA（`to_q/to_k/to_v/to_out.0`）+ 全量训练 `StateActionTokenizer` 与 `S0Encoder`
- **冻结**：VAE、T5 text encoder、DiT 主体权重

### 3.2 视觉 Token 化

1. VAE encode 整条 49 帧视频 → latent `(B, 13, C, H', W')`
2. 首帧 RGB 条件 latent 与 video latent 在 **通道维** concat（标准 I2V 做法）
3. `patch_embed` 将每帧 latent 切为 spatial patch：
   - 256×256 分辨率：latent 32×32，patch_size=2 → **16×16 = 256 patch/帧**
   - 480×640 分辨率：latent 60×80 → **30×40 = 1200 patch/帧**

### 3.3 维度配置（动态读取，不硬编码）

| 名称 | 来源 | CogVideoX-5B 典型值 | 用途 |
|------|------|---------------------|------|
| `hidden_dim` | `num_heads × head_dim` | 3072 | DiT 内部 token、SA token |
| `text_embed_dim` | `transformer.config.text_embed_dim` | 4096 | T5 prompt embedding、S0 条件 token |

`S0Encoder` 输出维度必须等于 `text_embed_dim`，以便拼接到 prompt embedding 后进入 cross-attention。

---

## 4. I2AV 扩展模块

### 4.1 S0Encoder — 初始状态全局条件

```
s0 (B, 7)  ──Linear(7→256→SiLU→Linear→4×D)──►  s0_cond (B, 4, D_text)
```

- 输入：归一化后的初始关节角 \(s_0\)
- 输出：4 个 **clean 条件 token**，拼接到 T5 text embedding 之后
- 不参与扩散加噪；所有后续 token 均可 attend 到条件区

### 4.2 StateActionTokenizer — 每帧 8 个 SA Token

每 latent 帧 8 token = **4 state + 4 action**：

```
state_t (7)  ──proj──► 4 × state_token  (+ state_modality_emb)
action_t (7) ──proj──► 4 × action_token (+ action_modality_emb)
                         └─ concat ─► (8, D) per frame
```

- **State token**：编码归一化绝对关节角，与视觉语义对齐（pose 帧是 \(s_t\) 的 FK 渲染）
- **Action token**：编码归一化增量 \(\Delta s_t\)，提供 MDP 决策语义
- 编解码对称：`encode()` 用于训练加噪前；`decode()` 从 DiT 输出还原 7DoF

参数量约 **12.8M**（约为 DiT 的 0.6%）。

---

## 5. 完整 Token 序列（49 帧 / 256×256 配置）

### 5.1 序列布局

```
┌─────────────── 条件区（全局可见，S0 不加噪）───────────────┐
│  Text (226)  │  S0_cond (4)  │                             │
└──────────────── 230 token ─────────────────────────────────┘

┌──────────── 视频 + 状态 + 动作区（temporal causal，联合去噪）────────────┐
│ F0(256) │ S0(4)+A0(4) │ F1(256) │ S1(4)+A1(4) │ ... │ F12(256) │ S12(4)+A12(4) │
└──────────────────────────── 3432 token ────────────────────────────────────┘

总长度 = 230 + 13 × (256 + 8) = 3662 token
```

每帧 chunk = **264 token** = 256 visual + 8 SA。

实现上，`forward_i2av_transformer()` 在 patch_embed 之后调用 `interleave_visual_sa_tokens()` 完成交错；输出前 `deinterleave_visual_sa_tokens()` 分离 visual 与 SA，visual 走 `proj_out`，SA 直接作为 loss 输入。

### 5.2 Time ID 分配

条件区 `time_id = -1`；视频区内交替偶数/奇数：

| Token 类型 | time_id | 数量/步 |
|-----------|---------|--------|
| Text, S0_cond | -1 | 230 |
| Frame_t (visual) | 2t | 256 |
| [State+Action]_t | 2t+1 | 8 |

共 **26 个** 有效时间位置（13 帧 × 2）。

---

## 6. 因果推理过程（MDP 语义）

设计目标是把扩散去噪过程对齐到 **部分可观测 MDP** 的单步循环：

```
F0 ──► [S0,A0] ──► F1 ──► [S1,A1] ──► F2 ──► ... ──► F12 ──► [S12,A12]
观察     状态+决策    结果     状态+决策    结果              观察      状态+决策
```

### 6.1 可见性规则

| Query → Key | 是否可见 | 含义 |
|-------------|---------|------|
| 条件区 → 任意 | ✓（条件区之间互看） | 文本指令 + 初始状态锚点 |
| 任意 → 条件区 | ✓ | 全局条件 |
| Frame_t → Frame\_{≤t} | ✓ | 当前/历史画面 |
| Frame_t → [S,A]_t | **✗** | **先观察，后知状态/动作** |
| Frame_t → [S,A]\_{<t} | ✓ | 已知历史决策 |
| [S,A]_t → Frame_t | ✓ | 基于当前画面确认状态 |
| [S,A]_t → [S,A]_t | ✓ | 状态与动作协同 |
| Frame\_{t+1} → [S,A]_t | ✓ | 下一帧受上一步决策影响 |

### 6.2 推理直觉

去噪一步时，模型按因果顺序"思考"：

1. 读入文本 + \(s_0\) 锚点
2. 生成/去噪第 0 帧视觉（pose+rgb latent）—— 仅依赖条件与历史
3. 看到第 0 帧后，去噪 \([S_0, A_0]\) —— 确认状态并给出动作
4. 第 1 帧视觉可以 attend 到 \([S_0, A_0]\)，依此类推

这与机器人 **observe → localize → act → observe** 的闭环一致，避免"未看先判"。

---

## 7. 因果 Mask 设计

### 7.1 形式化定义

记 `time_id(i)` 为 token \(i\) 的时间标识。允许 attend 当且仅当：

\[
\text{allow}(i, j) = \begin{cases}
\text{True} & j \in \text{条件区} \\
\text{True} & j \in \text{视频区} \land \text{time\_id}(j) \le \text{time\_id}(i) \\
\text{False} & \text{otherwise}
\end{cases}
\]

条件区 token **不能** attend 到视频区（条件不泄露未来）。

### 7.2 Chunked SDPA 实现

完整 \(3662 \times 3662\) mask 过大且无法走 FlashAttention 快路径。`CogVideoXI2AVCausalTemporalAttnProcessor2_0` 采用 **分块 SDPA**：

```python
# 伪代码 — 对应 causal_attention.py
for frame_idx in range(num_latent_frames):
    chunk = query[:, :, chunk_start : chunk_start + tokens_per_step]  # 264 tokens
    kv    = key/value[:, :, :chunk_end]                                # 条件 + 所有历史 chunk

    # 帧内 mask：visual query (前256行) 不能 attend 当前 SA key (后8列)
    allow[:patches_per_frame, sa_key_start:chunk_end] = False

    output_chunk = scaled_dot_product_attention(chunk, kv, kv, attn_mask=allow)
```

**块间**：chunk \(t\) 的 query 可看所有 \(\le t\) 的 chunk（标准 temporal causal）。

**块内**：visual 不能看同帧 SA；SA 可以看同帧 visual（通过 allow 矩阵默认 True，仅屏蔽 visual→SA 方向）。

环境变量 `COAF_CAUSAL_ATTENTION_BACKEND=flex` 可切换 FlexAttention 后端；默认 `chunked`。

---

## 8. RoPE 位置编码适配

CogVideoX 对 video token 使用 3D RoPE（时间 + 空间）。

I2AV 在每帧 256 个 visual RoPE 行之后，**复制该帧最后一个 visual patch 的 RoPE** 并 repeat 8 次，作为 SA token 的行：

```python
# expand_rope_for_i2av — i2av_sequence.py
cos_parts.append(freqs_cos[end - 1 : end].repeat(sa_per_frame, 1))
```

效果：

- SA token 与对应帧 visual **共享时间相位**（2t / 2t+1 区分在 attention mask，不在 RoPE 重复次数上再细分）
- SA token 在 spatial 维使用 **虚拟位置**（超出 16×16 grid），避免与 patch 空间坐标冲突

---

## 9. 训练流程：联合扩散

### 9.1 单步前向

```
1. VAE encode video → video_latents (B,13,C,H',W')
2. VAE encode 条件图 → image_latents，channel concat
3. T5 encode prompt → (B,226,D_text)
4. prepare_gt(state_24) → state_gt_13, action_gt_13, s0_norm
5. s0_cond = S0Encoder(s0_norm); prompt = cat(prompt, s0_cond)
6. clean_sa = StateActionTokenizer.encode(state_gt, action_gt)
7. 采样统一 timestep t；对 video_latent 与 clean_sa 加噪
8. interleave → DiT blocks (causal attn + RoPE) → deinterleave
9. video → velocity loss; SA → decode + MSE losses
```

**关键**：visual 与 SA 共享 **同一个** `timesteps`，保证联合去噪同步。

### 9.2 损失函数

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{video}} + \lambda_{sa} \cdot \mathcal{L}_{sa}
\]

\[
\mathcal{L}_{sa} = \lambda_s \mathcal{L}_{\text{state}} + \lambda_a \mathcal{L}_{\text{action}} + \lambda_c \mathcal{L}_{\text{consistency}}
\]

| 项 | 含义 |
|----|------|
| \(\mathcal{L}_{\text{video}}\) | CogVideoX velocity MSE（仅 visual latent） |
| \(\mathcal{L}_{\text{state}}\) | 预测 vs GT 绝对关节角 |
| \(\mathcal{L}_{\text{action}}\) | 预测 vs GT 动作增量 |
| \(\mathcal{L}_{\text{consistency}}\) | \(\| (s_{t+1}-s_t) - a_t \|\)，强制 state/action 自洽 |

默认：\(\lambda_{sa}=1, \lambda_s=1, \lambda_a=1, \lambda_c=0.5\)。

### 9.3 Checkpoint 内容

- LoRA 权重（DiT attention adapters）
- `state_action.pt`：`sa_tokenizer` + `s0_encoder` state dict
- `causal_attention.json`：序列长度、帧数、patch 数等元数据

---

## 10. 端到端数据流总览

```
                         ┌──────────────────────────────────────┐
  episode 数据            │  raw: rgb(25) + state(24) + action   │
                         │  modalities: pose/depth/flow(24)      │
                         └──────────────┬───────────────────────┘
                                        │ compose_all.py
                                        ▼
                         ┌──────────────────────────────────────┐
                         │  mp4: [modality×24 | rgb×25] = 49帧   │
                         │  cond_image: rgb[0]                   │
                         │  prompt: instruction                   │
                         └──────────────┬───────────────────────┘
                                        │ VAE (÷4 时间, ÷8 空间)
                                        ▼
                         ┌──────────────────────────────────────┐
                         │  13 latent frames × 256 patches      │
                         │  + 13 × 8 SA tokens (联合去噪)        │
                         │  + text(226) + s0(4) 条件            │
                         └──────────────┬───────────────────────┘
                                        │ Causal DiT + LoRA
                                        ▼
                         ┌──────────────────────────────────────┐
                         │  输出: 去噪 video latent               │
                         │       + 13-step state/action 轨迹      │
                         └──────────────────────────────────────┘
```

---

## 11. 与纯 Causal I2V 的差异

| 维度 | Causal I2V | Causal I2AV |
|------|-----------|-------------|
| Token 序列 | 226 + 13×256 = 3554 | 226 + 4 + 13×264 = **3662** |
| 条件 | Text + 首帧 RGB | Text + **S0** + 首帧 RGB |
| 去噪目标 | 仅 visual | visual + **state/action** |
| Attention | 帧级 causal | 帧级 causal + **帧内 visual↛SA** |
| 额外模块 | 无 | StateActionTokenizer, S0Encoder |
| Loss | \(\mathcal{L}_{video}\) | \(\mathcal{L}_{video} + \lambda_{sa}\mathcal{L}_{sa}\) |

---

## 12. 配置速查

### 12.1 常用超参（训练脚本）

```bash
--enable_i2av
--temporal_causal_attention
--state_norm_stats .../state_norm_stats.pt
--sa_per_frame 8          # 4 state + 4 action
--s0_cond_tokens 4
--lambda_sa 1.0 --lambda_s 1.0 --lambda_a 1.0 --lambda_c 0.5
--max_num_frames 49       # v5 用 73
--ignore_learned_positional_embeddings  # 非默认分辨率必须开启
```

### 12.2 序列长度公式

\[
L = L_{\text{text}} + L_{s0} + F_{\text{latent}} \times (P + K)
\]

- \(L_{\text{text}} = 226\)
- \(L_{s0} = 4\)
- \(F_{\text{latent}} = (F_{\text{pixel}}-1)/4 + 1\)
- \(P = (H/16) \times (W/16)\)（256 分辨率下 \(P=256\)）
- \(K = 8\)（SA token 数）

**示例**：49 帧、256×256 → \(226 + 4 + 13 \times 264 = 3662\)。

---

## 13. 实现注意事项

1. **维度**：`hidden_dim` 与 `text_embed_dim` 从 `transformer.config` 读取，适配不同 CogVideoX 变体。
2. **RoPE 长度**：`prepare_i2av_rotary_positional_embeddings` 从 base RoPE 反推 latent 帧数，避免对已是 latent 帧数的 `num_frames` 二次时间压缩。
3. **DDP**：访问 `sa_tokenizer` / `s0_encoder` 需 `accelerator.unwrap_model(i2av_aux)`。
4. **State 数据源**：manifest 默认指向 `coaf_dataset/raw/episode_*/state/state.npy`（24 帧），不依赖可能缺失的中间插值目录。
5. **推理**：完整 I2AV 采样循环（SA token 迭代去噪 + 解码轨迹）仍在 preview 阶段；当前 infer 脚本主要验证 visual + checkpoint 加载。

---

## 14. v5 Chunked I2AV 更新

v5 将 49 帧视频解释为 **25 reason + 24 RGB**。CogVideoX causal VAE 的时间压缩使 reason/RGB 边界落在 latent `L6|L7`，因此 visual token 被分为：

- `F_pose = (pose_pixel_frames - 1) // 4 + 1`
- `F_rgb = F_total - F_pose`
- `P = patches_per_latent_frame(...)`，由 `CogVideoXTransformer3D.config.patch_size`、VAE spatial scale 与训练分辨率动态计算

序列长度不再硬编码：

```text
L_cond = max_text_seq_length + s0_cond_tokens
L_video = F_pose * (P + chunk_token_count) + F_rgb * P
```

首帧 RGB 仍使用 CogVideoX-I2V 的通道 concat 条件：它不作为 self-attention 序列中的独立 token。

### 14.1 Joint Attention 而非 attn2

当前 diffusers 的 `CogVideoXBlock` 只有联合 `attn1`，没有独立 `attn2`。因此 Text/S0 与 video 的关系全部在 `attn1` processor 中处理：

| Query | 可见 Key | 因果性 |
|-------|----------|--------|
| Text/S0 | Text/S0 | 非因果，全互见 |
| Text/S0 | video | 不可见，防未来泄漏 |
| `P_k` | 条件 + 历史 pose/chunk + `P_k` | 因果，不可见同块 `c_k` |
| `c_k` | 条件 + 历史 pose/chunk + `P_k` + `c_k` | 块间因果，块内双向 |
| RGB 段 | 条件 + 全部 pose/chunk + 全部 RGB | 规划后渲染，RGB 段内非因果 |

实现入口：

- `i2av_layout.py`：`I2AVV5Layout` 与动态 layout 推导
- `i2av_forward.py`：`forward_i2av_v5_transformer`
- `causal_attention.py`：`CogVideoXI2AVV5CausalAttnProcessor2_0`
- `state_action.py`：`ChunkedStateActionTokenizer`

---

*文档版本：与 `Casual_CoAF/training/cog_video_training` 代码库同步，主配置为 CogVideoX-5B-I2V + v1/v4 49 帧训练。*
