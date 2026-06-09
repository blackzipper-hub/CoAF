# V5版本

# CoAF v5 实现计划 — 2524 帧划分  Action Chunking

> **基座模型**: CogVideoXI2V 5B causal temporal attention 已实现 **核心方案**: 25 帧 pose  24 帧 RGB，合并 VAE，state/action 以 action chunking 绑定 pose 段 **本版定位**: 通过帧数重新划分让 pose/RGB 边界对齐 VAE latent 边界，彻底消除边界混合  分布对不齐问题

---

## 一、版本演进与本版的关键改进

### 11 v5 的核心洞察

CogVideoX causal VAE 的时间分组固定:首帧单独,之后每 4 帧一组。latent 自然分界落在每个 4k 帧之后帧 01, 45, , 2425, 。

把 pose 设为 25 帧、RGB 设为 24 帧,pose/RGB 边界落在帧 2425,**正好卡在 L6 和 L7 之间**:

```Plain
L0 ← 帧 0          ┐
L1 ← 帧 1-4        │
L2 ← 帧 5-8        │
L3 ← 帧 9-12       ├─ Pose 段（7 个干净 latent L0-L6）
L4 ← 帧 13-16      │
L5 ← 帧 17-20      │
L6 ← 帧 21-24      ┘
─────────────────────  ← 边界正好在这里
L7 ← 帧 25-28      ┐
L8 ← 帧 29-32      │
L9 ← 帧 33-36      ├─ RGB 段（6 个干净 latent L7-L12）
L10 ← 帧 37-40     │
L11 ← 帧 41-44     │
L12 ← 帧 45-48     ┘
```

这一个改动同时解决:

- **合并 VAE 即可**不需分开,无分布对不齐风险
- **边界干净**pose 和 RGB 在 latent 层面天然分开,无混合 latent
- **chunk 分配整齐**25 个 state 值对应 7 个 pose latent

---

## 二、最终架构

### 21 序列结构

```Plain
┌──── 条件区(全局可见,不去噪)────┐
│  Text (226) │ S₀ (4)          │RGB图片
└──────── 230 token ─────────────┘

┌─────── Pose 段（action chunking, chunk 间因果）───────┐  ┌──── RGB 段（渲染）────┐
│ P0 [c0] P1 [c1] P2 [c2] ... P6 [c6]                   │  │ R0 R1 R2 R3 R4 R5    │
│ 观测+动作块交替                                          │  │ 真实画面渲染            │
└────────── 7 pose latent + 7 chunks ───────────────────┘  └─── 6 RGB latent ─────┘
```

### 22 Chunk 与轨迹对齐按 VAE 实际时间映射

```Plain
Pose latent   覆盖像素帧    chunk 内容（state+action 交错）
──────────────────────────────────────────────────────────
P0 (L0)       帧 0          [s0 a0]                    （1 步，首帧单独）
P1 (L1)       帧 1-4        [s1 a1 s2 a2 s3 a3 s4 a4]  （4 步）
P2 (L2)       帧 5-8        [s5 a5 ... s8 a8]          （4 步）
P3 (L3)       帧 9-12       [s9 a9 ... s12 a12]
P4 (L4)       帧 13-16      [s13 a13 ... s16 a16]
P5 (L5)       帧 17-20      [s17 a17 ... s20 a20]
P6 (L6)       帧 21-24      [s21 a21 ... s24 a24]
──────────────────────────────────────────────────────────
合计: 1 + 4×6 = 25 步（s0-s24），与 25 帧 pose 一一对应
```

轨迹是 25 步s0s24,正好对应 25 帧 pose、7 个 pose latent。每个 chunk 内容严格对应该 latent 在画面上展示的运动。

### 23 因果可见性chunk 内双向  chunk 间因果

关键语义:

- Pose Pk 看不到自己的 chunk ck → 先观测,后产出动作
- chunk ck 内部双向 → 一次规划一小段（处理 2）
- chunk ck 看不到下一个观测 Pk1 → 跨 chunk 因果
- RGB 段看到完整 pose  state  action → 渲染基于完整规划

---

## 三、数据准备

### 31 帧划分与 pose 渲染

```Python
def prepare_video(trajectory):
    """
    构造 25 帧 pose + 24 帧 RGB = 49 帧拼接视频。
    
    pose: 用 BridgeV2 state 序列经 FK 渲染（25 帧）
    rgb:  真实 RGB 序列（24 帧）
    """
    # state 序列取 25 步（s0-s24）
    states_25 = trajectory['observation/state'][:25]   # (25, 7)
    
    # FK 渲染成 25 帧 pose silhouette
    pose_frames = fk_render(states_25)                 # (25, 3, 256, 256)
    
    # RGB 取 24 帧（与 pose 同步采样，覆盖同一段运动）
    rgb_frames = trajectory['rgb'][:24]                # (24, 3, 256, 256)
    
    # 拼接：pose 在前，RGB 在后
    video_49 = torch.cat([pose_frames, rgb_frames], dim=0)  # (49, 3, 256, 256)
    return video_49, states_25
```

注: pose 和 RGB 都覆盖同一段运动轨迹（024），只是 RGB 少采样 1 帧（24 vs 25）。 RGB 段不需要和 state 对齐（state 只绑 pose 段），所以 RGB 24 帧无碍。

### 32 VAE 时间映射（动手前必须实测确认）

```Python
# 用边界清晰的测试视频确认 VAE 分组
test = torch.cat([
    torch.ones(1, 3, 25, 256, 256),   # 前 25 帧白（模拟 pose）
    torch.zeros(1, 3, 24, 256, 256),  # 后 24 帧黑（模拟 RGB）
], dim=2)

latent = vae.encode(test).latent_dist.sample()
print("latent 时间维:", latent.shape[2])   # 期望 13

decoded = vae.decode(latent).sample
# 检查 decode 后帧 24（应纯白）和帧 25（应纯黑）是否干净
# 若边界帧无灰色混合 → 确认 L6|L7 是干净分界
# 若有混合 → VAE 实际分组与推算不同，需调整帧数划分
```

### 33 VAE 时间映射表

```Python
# 确认后写死这个映射（标准 CogVideoX causal VAE）
POSE_LATENT_MAP = [
    (0, [0]),                  # L0 ← 帧 0
    (1, [1, 2, 3, 4]),         # L1 ← 帧 1-4
    (2, [5, 6, 7, 8]),         # L2 ← 帧 5-8
    (3, [9, 10, 11, 12]),      # L3 ← 帧 9-12
    (4, [13, 14, 15, 16]),     # L4 ← 帧 13-16
    (5, [17, 18, 19, 20]),     # L5 ← 帧 17-20
    (6, [21, 22, 23, 24]),     # L6 ← 帧 21-24
]
# RGB 段 L7-L12 不挂 chunk
```

### 34 归一化统计量

```Python
def compute_norm_stats(dataset):
    all_states = np.concatenate(
        [traj['observation/state'] for traj in dataset], axis=0
    )  # (N, 7)
    stats = {
        'mean': torch.tensor(all_states.mean(0), dtype=torch.float32),
        'std': torch.tensor(all_states.std(0), dtype=torch.float32).clamp(min=1e-6),
    }
    torch.save(stats, 'state_norm_stats.pt')
    return stats
```

### 35 State/Action GT 构造

```Python
def prepare_gt(states_25, norm_stats):
    """
    Args:
        states_25: (B, 25, 7)  s0-s24 绝对关节角
    Returns:
        state_gt:  (B, 25, 7)  归一化绝对关节角
        action_gt: (B, 25, 7)  归一化增量（末步 pad 0）
        s0_norm:   (B, 7)
    """
    mean, std = norm_stats['mean'], norm_stats['std']
    
    state_gt = (states_25 - mean) / std                       # 绝对值归一化
    
    delta = states_25[:, 1:] - states_25[:, :-1]              # (B, 24, 7)
    delta = F.pad(delta, (0, 0, 0, 1), value=0)              # (B, 25, 7)
    action_gt = delta / std                                   # delta 只除 std
    
    s0_norm = state_gt[:, 0]                                  # (B, 7)
    return state_gt, action_gt, s0_norm
```

---

## 四、新增代码模块

### 41 ChunkedStateActionTokenizer

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChunkedStateActionTokenizer(nn.Module):
    """
    将 25 步轨迹按 VAE 映射切成 7 个 chunk，每 chunk 含若干步 state+action。
    chunk 内 state/action 交错: [s a s a ...]
    
    采用处理 B: 首 chunk pad 到 4 步，所有 chunk 统一 4 步（布局规则）。
    """
    def __init__(self, state_dim=7, hidden_dim=3072,
                 num_chunks=7, steps_per_chunk=4):
        super().__init__()
        self.num_chunks = num_chunks
        self.steps_per_chunk = steps_per_chunk
        self.hidden_dim = hidden_dim
        # 每 chunk token 数 = steps × 2 (state + action)
        self.chunk_token_count = steps_per_chunk * 2
        
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, 256), nn.SiLU(), nn.Linear(256, hidden_dim)
        )
        self.action_proj = nn.Sequential(
            nn.Linear(state_dim, 256), nn.SiLU(), nn.Linear(256, hidden_dim)
        )
        self.state_output = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.SiLU(), nn.Linear(256, state_dim)
        )
        self.action_output = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.SiLU(), nn.Linear(256, state_dim)
        )
        self.state_modality = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.action_modality = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
    
    def _pad_to_chunks(self, seq):
        """
        把 25 步序列按 VAE 映射 pad 成 7×4=28 步的规则布局。
        首 chunk（1 步）pad 到 4 步（重复首步）。
        映射: [0] | [1-4] | [5-8] | ... | [21-24]
        pad 后: [0,0,0,0] | [1,2,3,4] | ... | [21,22,23,24]
        """
        B, T, D = seq.shape  # T=25
        # 首步重复 3 次补到 4
        first = seq[:, 0:1].repeat(1, 3, 1)   # (B, 3, D) 重复 s0
        padded = torch.cat([first, seq], dim=1)  # (B, 28, D)
        return padded  # 28 = 7 chunks × 4 steps
    
    def encode(self, state_norm, action_norm):
        """
        state_norm, action_norm: (B, 25, 7)
        Returns: (B, num_chunks, chunk_token_count, D) = (B, 7, 8, D)
        """
        B = state_norm.shape[0]
        s_pad = self._pad_to_chunks(state_norm)   # (B, 28, 7)
        a_pad = self._pad_to_chunks(action_norm)  # (B, 28, 7)
        
        s_tok = self.state_proj(s_pad) + self.state_modality    # (B, 28, D)
        a_tok = self.action_proj(a_pad) + self.action_modality  # (B, 28, D)
        
        # 交错: [s0 a0 s1 a1 ...]
        inter = torch.stack([s_tok, a_tok], dim=2)              # (B, 28, 2, D)
        inter = inter.reshape(B, 56, self.hidden_dim)           # (B, 56, D)
        
        # 切 chunk: 56 = 7 × 8
        chunks = inter.reshape(B, self.num_chunks,
                               self.chunk_token_count, self.hidden_dim)
        return chunks  # (B, 7, 8, D)
    
    def decode(self, chunk_out):
        """
        chunk_out: (B, 7, 8, D)
        Returns: pred_state (B, 25, 7), pred_action (B, 25, 7)
        （去掉首 chunk 的 3 个 pad 步）
        """
        B = chunk_out.shape[0]
        flat = chunk_out.reshape(B, 56, self.hidden_dim)        # (B, 56, D)
        flat = flat.reshape(B, 28, 2, self.hidden_dim)          # (B, 28, 2, D)
        s_tok = flat[:, :, 0]                                   # (B, 28, D)
        a_tok = flat[:, :, 1]                                   # (B, 28, D)
        
        pred_state_28 = self.state_output(s_tok)                # (B, 28, 7)
        pred_action_28 = self.action_output(a_tok)              # (B, 28, 7)
        
        # 去掉首 chunk 的 3 个 pad 步（保留第 3 个即真实 s0）
        pred_state = pred_state_28[:, 3:]                       # (B, 25, 7)
        pred_action = pred_action_28[:, 3:]                     # (B, 25, 7)
        return pred_state, pred_action
```

### 42 S0Encoder

```Python
class S0Encoder(nn.Module):
    def __init__(self, state_dim=7, hidden_dim=3072, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(state_dim, 256), nn.SiLU(),
            nn.Linear(256, num_tokens * hidden_dim)
        )
    def forward(self, s0_norm):
        B = s0_norm.shape[0]
        return self.proj(s0_norm).reshape(B, self.num_tokens, -1)
```

### 43 StateActionAttentionBias（可选，防淹没）

```Python
class StateActionAttentionBias(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_heads))
    def apply(self, attn_logits, chunk_key_mask):
        bias = self.bias.view(1, -1, 1, 1)
        attn_logits[:, :, :, chunk_key_mask] += bias
        return attn_logits
```

---

## 五、Token 序列构造

```Python
def build_full_sequence(text_tok, s0_tok, pose_tok, chunks, rgb_tok):
    """
    Args:
        text_tok: (B, 226, D)
        s0_tok:   (B, 4, D)
        pose_tok: (B, 7, 256, D)   7 pose latent
        chunks:   (B, 7, 8, D)     7 chunk × 8 token
        rgb_tok:  (B, 6, 256, D)   6 RGB latent
    Returns:
        sequence: (B, N, D)
        meta:     dict
    """
    B = text_tok.shape[0]
    
    condition = torch.cat([text_tok, s0_tok], dim=1)   # (B, 230, D)
    
    # Pose 段: P0 c0 P1 c1 ... P6 c6
    pose_parts = []
    for k in range(7):
        pose_parts.append(pose_tok[:, k])    # (B, 256, D)
        pose_parts.append(chunks[:, k])      # (B, 8, D)
    pose_seg = torch.cat(pose_parts, dim=1)  # (B, 7×264 = 1848, D)
    
    # RGB 段
    rgb_seg = rgb_tok.reshape(B, 6 * 256, -1)  # (B, 1536, D)
    
    sequence = torch.cat([condition, pose_seg, rgb_seg], dim=1)
    # 总长 = 230 + 1848 + 1536 = 3614
    
    # 元信息
    meta = {'condition': (0, 230), 'pose': {}, 'chunk': {}, 'rgb': {}}
    offset = 230
    for k in range(7):
        meta['pose'][k] = (offset, offset + 256); offset += 256
        meta['chunk'][k] = (offset, offset + 8); offset += 8
    for j in range(6):
        meta['rgb'][j] = (offset, offset + 256); offset += 256
    
    return sequence, meta


def extract_outputs(output, meta):
    """分离 pose / chunk / rgb 三部分"""
    pose_parts, chunk_parts, rgb_parts = [], [], []
    for k in range(7):
        ps, pe = meta['pose'][k];  pose_parts.append(output[:, ps:pe])
        cs, ce = meta['chunk'][k]; chunk_parts.append(output[:, cs:ce])
    for j in range(6):
        rs, re = meta['rgb'][j];   rgb_parts.append(output[:, rs:re])
    
    pose_out = torch.stack(pose_parts, dim=1)    # (B, 7, 256, D)
    chunk_out = torch.stack(chunk_parts, dim=1)  # (B, 7, 8, D)
    rgb_out = torch.stack(rgb_parts, dim=1)      # (B, 6, 256, D)
    return pose_out, chunk_out, rgb_out
```

---

## 六、Causal Mask

### 61 Time ID

```Python
def build_time_ids():
    """
    条件区:    -1
    Pose P_k:  2k       （观测）
    chunk c_k: 2k+1     （整 chunk 共享 → 内部双向）
    RGB R_j:   14+j     （排在所有 pose/chunk 之后）
    """
    time_ids = []
    time_ids.extend([-1] * 230)                  # 条件区
    for k in range(7):
        time_ids.extend([2 * k] * 256)           # P_k
        time_ids.extend([2 * k + 1] * 8)         # c_k（共享 time_id）
    for j in range(6):
        time_ids.extend([14 + j] * 256)          # R_j
    return torch.tensor(time_ids)                # (3614,)
```

### 62 Mask 构造

```Python
def build_causal_mask(time_ids):
    N = len(time_ids)
    is_cond = (time_ids == -1)
    mask = time_ids.unsqueeze(0) >= time_ids.unsqueeze(1)
    mask[:, is_cond] = True
    cond_idx = is_cond.nonzero(as_tuple=True)[0]
    non_cond_idx = (~is_cond).nonzero(as_tuple=True)[0]
    mask[cond_idx.unsqueeze(1), non_cond_idx.unsqueeze(0)] = False
    return mask
```

### 63 验证

```Python
def verify_mask(mask, meta):
    c0 = meta['chunk'][0][0]; p0 = meta['pose'][0][0]
    p1 = meta['pose'][1][0]; r0 = meta['rgb'][0][0]
    
    assert not mask[p0, c0],          "P0 不应看到 c0（先观测）"
    assert mask[c0, p0],              "c0 应看到 P0（基于观测决策）"
    assert mask[c0, c0 + 7],          "chunk 内双向（s0 看 a3）"
    assert mask[c0 + 7, c0],          "chunk 内双向（a3 看 s0）"
    assert not mask[c0, p1],          "c0 不应看到 P1（跨 chunk 因果）"
    assert mask[r0, meta['chunk'][6][0]], "RGB 应看到全部 chunk"
    assert mask[meta['pose'][3][0], meta['chunk'][2][0]], "P3 应看到 c2"
    print("✓ 因果语义验证通过")
```

### 64 Chunked SDPA 适配

```Python
def chunked_causal_sdpa(hidden, condition_kv):
    """
    Pose 段: 每对 (P_k 256 + c_k 8) = 264 token 为一个 chunk，7 次循环
    RGB 段:  6 个 latent，可一次性算（都能看全部 pose + 过去 RGB）
    """
    outputs = []
    kv = condition_kv  # (B, 230, D)
    
    # Pose 段循环
    offset = 230
    for k in range(7):
        block = hidden[:, offset:offset + 264]      # P_k + c_k
        cur_kv = torch.cat([kv, block], dim=1)
        len_kv = cur_kv.shape[1]
        # 块内 mask: 前 256（P_k）不能看后 8（c_k）
        inner = torch.ones(264, len_kv, dtype=torch.bool, device=hidden.device)
        inner[:256, -8:] = False
        out = F.scaled_dot_product_attention(block, cur_kv, cur_kv, attn_mask=inner)
        outputs.append(out)
        kv = cur_kv
        offset += 264
    
    # RGB 段：能看全部 pose + chunk + 过去 RGB
    rgb_block = hidden[:, offset:]                   # (B, 6×256, D)
    rgb_kv = torch.cat([kv, rgb_block], dim=1)
    # RGB 段内部 causal（R_j 看 R0..R_j），用标准下三角
    rgb_out = F.scaled_dot_product_attention(rgb_block, rgb_kv, rgb_kv, is_causal=False)
    # 注: RGB 段内部若需严格 causal，需另构 mask；此处简化为可见全部 pose+过去RGB
    outputs.append(rgb_out)
    
    return torch.cat(outputs, dim=1)
```

---

## 七、位置编码

```Python
def build_rope_ids():
    """与 time_id 同结构：pose/chunk 交替占 0-13，RGB 占 14-19"""
    rope = []
    for k in range(7):
        rope.extend([2 * k] * 256)
        rope.extend([2 * k + 1] * 8)
    for j in range(6):
        rope.extend([14 + j] * 256)
    return torch.tensor(rope)
    # 确认 CogVideoX RoPE 表最大长度 >= 20

def build_spatial_ids():
    """Pose/RGB patch: (row,col) 16×16; chunk token: 虚拟位置 (16+i, 0)"""
    sids = []
    for k in range(7):
        for r in range(16):
            for c in range(16): sids.append((r, c))
        for i in range(8): sids.append((16 + i, 0))
    for j in range(6):
        for r in range(16):
            for c in range(16): sids.append((r, c))
    return sids
```

---

## 八、三阶段训练

### 阶段 1: chunk = clean GT, 训视频去噪

```Python
def train_stage1(dit, sa_tok, s0_enc, batch, scheduler, opt):
    video_49, states_25 = batch['video'], batch['states']
    latent = vae.encode(video_49).latent_dist.sample()       # (B, C, 13, h, w)
    pose_tok = patchify(latent[:, :, :7])                    # (B, 7, 256, D)
    rgb_tok = patchify(latent[:, :, 7:])                     # (B, 6, 256, D)
    
    state_gt, action_gt, s0_norm = prepare_gt(states_25, norm_stats)
    text_tok = text_encoder(batch['text'])
    s0_tok = s0_enc(s0_norm)
    chunks = sa_tok.encode(state_gt, action_gt)              # clean，不加噪
    
    t = torch.randint(0, 1000, (B,))
    npose = torch.randn_like(pose_tok); nrgb = torch.randn_like(rgb_tok)
    pose_noisy = scheduler.add_noise(pose_tok, npose, t)
    rgb_noisy = scheduler.add_noise(rgb_tok, nrgb, t)
    
    seq, meta = build_full_sequence(text_tok, s0_tok, pose_noisy, chunks, rgb_noisy)
    out = dit(seq, t, causal_mask)
    pose_out, _, rgb_out = extract_outputs(out, meta)
    
    L_video = F.mse_loss(pose_out, npose) + F.mse_loss(rgb_out, nrgb)
    opt.zero_grad(); L_video.backward(); opt.step()
    return {'L_video': L_video.item()}
```

配置: DiT LoRAlr 1e4  satokencode  s0enc 训练; satokdecode 冻结。 步数 30005000。后 500 步对 clean chunk 加 10% 噪声缓解 exposure bias。

### 阶段 2: 视觉 clean, 训 chunk 去噪

```Python
def train_stage2(dit, sa_tok, s0_enc, batch, scheduler, opt):
    with torch.no_grad():
        latent = vae.encode(batch['video']).latent_dist.sample()
        pose_tok = patchify(latent[:, :, :7])
        rgb_tok = patchify(latent[:, :, 7:])
        text_tok = text_encoder(batch['text'])
        s0_tok = s0_enc(s0_norm)
    
    state_gt, action_gt, _ = prepare_gt(batch['states'], norm_stats)
    chunks = sa_tok.encode(state_gt, action_gt)
    nchunk = torch.randn_like(chunks); t = torch.randint(0, 1000, (B,))
    chunks_noisy = scheduler.add_noise(chunks, nchunk, t)
    
    seq, meta = build_full_sequence(text_tok, s0_tok, pose_tok, chunks_noisy, rgb_tok)
    out = dit(seq, t, causal_mask)
    _, chunk_out, _ = extract_outputs(out, meta)
    
    pred_state, pred_action = sa_tok.decode(chunk_out)
    L = compute_sa_loss(pred_state, pred_action, state_gt, action_gt)
    opt.zero_grad(); L.backward(); opt.step()
    return {'L': L.item()}
```

配置: DiT 全冻结; 只训 satok含 decode。步数 10002000。

### 阶段 3: 联合去噪

```Python
def train_stage3(dit, sa_tok, s0_enc, batch, scheduler, opt):
    latent = vae.encode(batch['video']).latent_dist.sample()
    pose_tok = patchify(latent[:, :, :7]); rgb_tok = patchify(latent[:, :, 7:])
    state_gt, action_gt, s0_norm = prepare_gt(batch['states'], norm_stats)
    chunks = sa_tok.encode(state_gt, action_gt)
    text_tok = text_encoder(batch['text']); s0_tok = s0_enc(s0_norm)
    
    t = torch.randint(0, 1000, (B,))
    npose = torch.randn_like(pose_tok); nrgb = torch.randn_like(rgb_tok)
    nchunk = torch.randn_like(chunks)
    pose_noisy = scheduler.add_noise(pose_tok, npose, t)
    rgb_noisy = scheduler.add_noise(rgb_tok, nrgb, t)
    chunks_noisy = scheduler.add_noise(chunks, nchunk, t)
    
    seq, meta = build_full_sequence(text_tok, s0_tok, pose_noisy, chunks_noisy, rgb_noisy)
    out = dit(seq, t, causal_mask)
    pose_out, chunk_out, rgb_out = extract_outputs(out, meta)
    
    L_video = F.mse_loss(pose_out, npose) + F.mse_loss(rgb_out, nrgb)
    pred_state, pred_action = sa_tok.decode(chunk_out)
    L_sa = compute_sa_loss(pred_state, pred_action, state_gt, action_gt)
    L_total = L_video + 0.1 * L_sa
    
    opt.zero_grad(); L_total.backward(); opt.step()
    return {'L_video': L_video.item(), 'L_sa': L_sa.item()}
```

配置: 全部解冻,DiT lr = 1e5阶段 1 的 1/10。步数 20003000。

### Loss

```Python
def compute_sa_loss(pred_state, pred_action, state_gt, action_gt,
                    ls=1.0, la=1.0, lc=0.5):
    L_state = F.mse_loss(pred_state, state_gt)
    L_action = F.mse_loss(pred_action, action_gt)
    implied = pred_state[:, 1:] - pred_state[:, :-1]
    L_consistency = F.mse_loss(implied, pred_action[:, :-1])
    return ls * L_state + la * L_action + lc * L_consistency
```

---

## 九、推理

```Python
@torch.no_grad()
def inference(dit, sa_tok, s0_enc, text, first_rgb, s0, scheduler, norm_stats):
    text_tok = text_encoder(text)
    s0_norm = (s0 - norm_stats['mean']) / norm_stats['std']
    s0_tok = s0_enc(s0_norm)
    cond = prepare_i2v_condition(first_rgb)
    
    # 从纯噪声开始
    pose_noisy = torch.randn(1, 7, 256, D)
    rgb_noisy = torch.randn(1, 6, 256, D)
    chunks_noisy = torch.randn(1, 7, 8, D)
    
    causal_mask = build_causal_mask(build_time_ids())
    
    for t in scheduler.timesteps:
        seq, meta = build_full_sequence(text_tok, s0_tok, pose_noisy, chunks_noisy, rgb_noisy)
        out = dit(seq, t, causal_mask, cond)
        pose_p, chunk_p, rgb_p = extract_outputs(out, meta)
        pose_noisy = scheduler.step(pose_p, t, pose_noisy)
        rgb_noisy = scheduler.step(rgb_p, t, rgb_noisy)
        chunks_noisy = scheduler.step(chunk_p, t, chunks_noisy)
    
    # 解码视频
    full_latent = torch.cat([unpatchify(pose_noisy), unpatchify(rgb_noisy)], dim=2)
    video = vae.decode(full_latent).sample        # (1, 3, 49, 256, 256)
    pose_video = video[:, :, :25]                 # 25 pose 帧
    rgb_video = video[:, :, 25:]                  # 24 RGB 帧
    
    # 解码 state/action
    pred_state, pred_action = sa_tok.decode(chunks_noisy)   # (1, 25, 7)
    pred_state = pred_state * norm_stats['std'] + norm_stats['mean']
    pred_action = pred_action * norm_stats['std']
    
    # action delta（发控制器）: 用 state 差分更稳定
    action_delta = pred_state[:, 1:] - pred_state[:, :-1]   # (1, 24, 7)
    
    return pose_video, rgb_video, pred_state, action_delta
```

---

## 十、验证检查点

### 阶段 1 后

- 视频质量 ≥ 无 state/action 的 causal 版本
- 扰动 chunk GT → 视频变化（chunk 有影响）
- 扰动 S₀ → pose 起始姿态变化

### 阶段 2 后

- State / Action MSE 优于之前 MLP head
- Lconsistency 接近 0
- chunk 解码 state → FK 渲染 → 和 GT pose 大致吻合
- chunk 边界运动平滑

### 阶段 3 后

- 视频质量不退化
- State/Action 精度持平或提升
- **VideoState 一致性**: FKpredstate vs 生成 pose 段高度一致（核心指标）
- 端到端: pose、rgb、state、action 四者物理一致

### 消融（论文用）

这一部分的代码不需要完成

---

## 十二、风险与缓解

---

## 附:v5 一句话总结

把 pose 设为 25 帧、RGB 设为 24 帧,让 pose/RGB 边界正好落在 CogVideoX VAE 的 L6L7 latent 分界上——这样合并 VAE 就能得到 7 个干净 pose latent  6 个干净 RGB latent,无边界混合、无分布对不齐。

State/action 以 action chunking 绑定 7 个 pose latent按 VAE 时间映射分配,首 chunk 1 步、其余 4 步,共 25 步,chunk 内双向、chunk 间因果,实现 chunked MDP。

依然先按照action 和video联合降噪的逻辑来

最后再尝试三阶段训练的效果会不会更好

三阶段训练:clean chunk 训视频 → clean 视觉训 chunk → 联合去噪。

理论上得用深度图最好，因为输入的7dof是三维度的；

在深度图的基础上增加一个73帧的效果