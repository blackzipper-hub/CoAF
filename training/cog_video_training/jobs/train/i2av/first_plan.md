方案 A：Action token 插入每帧
每个 latent 时间步，在 256 个 spatial patch 之后追加 1 个 action token。这个 token 的干净值是 GT state 经过 Linear(7, 3072) 投影得到的 embedding。训练时按扩散 schedule 加噪，和视觉 token 一起去噪。
序列变成：
[ Text(226) | L0: 256 patch + 1 action | L1: 256 patch + 1 action | ... | L12: 256 patch + 1 action ]
总长 = 226 + 13×257 = 3567 token
Causal 约束自然适用——action_t 和 pose_t 处于同一时间步，互相可见；action_t 能看到 action_0 到 action_t，看不到 action_{t+1} 及之后。去噪结束后 action token 通过 Linear(3072, 7) 解码回关节角。
优势在于架构改动最小——只是在每帧的 token 末尾多插一个，causal mask 的逻辑完全不用改（新 token 和同帧的 spatial patch 共享时间步 id）。DiT 在生成视频时能直接 attend 到同一时刻的 action，实现了真正的"video 和 action 互相参考"。
风险是 action token 只有 1 个（7 维信息压缩到一个 3072 维 embedding），信息密度和 256 个视觉 patch 差距悬殊，attention 可能会忽略它。缓解方式是给 action token 加一个可学习的 scale factor，或者增加到 2-4 个 action token per frame。

详细计划
总体模型结构构造
1. Casual 调整、注入Action 轨迹，输出增加 7DoF 信息
2. 输入条件增加起始的 state 位置 / 
1.1 设计哲学
模型同时去噪三种信号：视觉帧（pose + RGB）、状态轨迹（绝对 7DoF）、动作序列（7DoF 增量）。 三者在同一个 DiT 的 self-attention 中交互，共享同一个扩散过程。
为什么同时要 state 和 action：
- State（绝对关节角）：与视觉 token 语义直接对齐（Frame_t 是 s_t 的 FK 渲染），每个 token 自包含，去噪不依赖累积
- Action（增量）：提供 MDP 决策语义（"该做什么"），分布紧凑（近零均值），是机器人控制器需要的信号
- 两者结合的独特价值：L_consistency 强制 state 和 action 自洽（s_{t+1} - s_t = a_t），同一条轨迹从两个视角互相约束，这一点在开始脚本构建之前，需要先读取几条case，并进行这样的验证
1.2 序列结构
CogVideoX-5B I2V
┌──── 条件区（全局可见，不参与去噪）─────┐
│  Text (226 tok)  │  S₀_cond (4 tok)  │
│  任务指令          │  初始关节角(锚点)
    初始的RGB图片   │
└─────────── 230 token ─────────────────┘

┌───── 视频+状态+动作区（temporal causal）───────────────────────────────────────────┐
│ F0(256) │ S0(4)+A0(4) │ F1(256) │ S1(4)+A1(4) │ ... │ F12(256) │ S12(4)+A12(4) │
│ visual    state action    visual    state action          visual     state  action  │
└──────────────────────────── 3432 token ───────────────────────────────────────────┘

总计 3662 token（原 3554，增加 3%）
每帧 8 个非视觉 token 的内部结构：hidden_dim
这里相当于做了个表征升维：
[ St_0  St_1  St_2  St_3 | At_0  At_1  At_2  At_3 ]
  ←─ 4 个 state token ─→   ←─ 4 个 action token ─→
  编码 s_t 绝对关节角         编码 a_t = s_{t+1} - s_t
暂时无法在飞书文档外展示此内容
1.4 因果可见性（完整 MDP）
F0 → [S0,A0] → F1 → [S1,A1] → F2 → [S2,A2] → ... → F12 → [S12,A12]
观察   状态+决策   结果   状态+决策   结果   状态+决策          观察    状态+决策
暂时无法在飞书文档外展示此内容
MDP 语义：
- Frame_t 看不到 [S,A]_t → 先观察，后确认状态/做决策
- [S,A]_t 能看到 Frame_t → 基于观察确认状态并决定动作
- Frame_{t+1} 能看到 [S,A]_t → 下一帧知道上一步的状态和动作
- State 和 Action token 在同一时间步内互相可见 → 状态感知和决策协同
1.5 三阶段训练策略
先不用按照三阶段，直接按照阶段3构建训练代码
暂时无法在飞书文档外展示此内容
损失设计
v3 的核心优势：三项 loss 互相约束。
def compute_sa_loss(sa_output, state_tokenizer, state_gt_13, action_gt_13,
                    lambda_s=1.0, lambda_a=1.0, lambda_c=0.5):
    """
    计算 State + Action + Consistency 三项 loss。
    
    Args:
        sa_output:     (B, 13, 8, D) DiT 输出的 state+action token
        state_tokenizer: StateActionTokenizer 实例
        state_gt_13:   (B, 13, 7)    GT 绝对关节角
        action_gt_13:  (B, 13, 7)    GT 增量
    
    Returns:
        loss_dict: dict 各项 loss
    """
    # 解码
    pred_state, pred_action = state_tokenizer.decode(sa_output)
    # pred_state:  (B, 13, 7) 预测的绝对关节角
    # pred_action: (B, 13, 7) 预测的动作增量
    
    # L_state: 绝对位置准确
    L_state = F.mse_loss(pred_state, state_gt_13)
    
    # L_action: 动作增量准确
    L_action = F.mse_loss(pred_action, action_gt_13)
    
    # L_consistency: state 和 action 互相一致
    # 从 state 推出的隐含 delta 应该等于直接预测的 action
    implied_delta = pred_state[:, 1:] - pred_state[:, :-1]       # (B, 12, 7)
    predicted_delta = pred_action[:, :-1]                         # (B, 12, 7)
    L_consistency = F.mse_loss(implied_delta, predicted_delta)
    
    # 加权总和
    L_sa = lambda_s * L_state + lambda_a * L_action + lambda_c * L_consistency
    
    return {
        'L_state': L_state,
        'L_action': L_action,
        'L_consistency': L_consistency,
        'L_sa': L_sa,
    }
暂时无法在飞书文档外展示此内容

---

---
二、新增代码模块
2.1 StateActionTokenizer
同时编码绝对状态和动作增量，解码时分别输出两者。
import torch
import torch.nn as nn

class StateActionTokenizer(nn.Module):
    """
    将 7DoF 绝对关节角 + 7DoF 动作增量编码成 K 个 token 参与 DiT 联合去噪。
    
    每帧 8 token = 4 state token + 4 action token
    
    编码: state(B,13,7) + action(B,13,7) → tokens(B,13,8,D)
    解码: tokens(B,13,8,D) → state(B,13,7) + action(B,13,7)
    """
    def __init__(self, state_dim=7, hidden_dim=3072,
                 num_state_tokens=4, num_action_tokens=4):
        super().__init__()
        self.num_state_tokens = num_state_tokens
        self.num_action_tokens = num_action_tokens
        self.num_tokens = num_state_tokens + num_action_tokens  # 8
        self.hidden_dim = hidden_dim
        
        # ===== 编码器 =====
        # State 编码: 绝对关节角(7) → 4 个 token
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, num_state_tokens * hidden_dim),
        )
        
        # Action 编码: delta 增量(7) → 4 个 token
        self.action_proj = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, num_action_tokens * hidden_dim),
        )
        
        # ===== 解码器 =====
        # State 解码: 4 token → 绝对关节角(7)
        self.state_output = nn.Sequential(
            nn.Linear(num_state_tokens * hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, state_dim),
        )
        
        # Action 解码: 4 token → delta 增量(7)
        self.action_output = nn.Sequential(
            nn.Linear(num_action_tokens * hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, state_dim),
        )
        
        # ===== 模态区分 =====
        self.state_modality_emb = nn.Parameter(
            torch.randn(1, 1, num_state_tokens, hidden_dim) * 0.02
        )
        self.action_modality_emb = nn.Parameter(
            torch.randn(1, 1, num_action_tokens, hidden_dim) * 0.02
        )
    
    def encode(self, state_norm, action_norm):
        """
        Args:
            state_norm:  (B, T, 7) 归一化的绝对关节角
            action_norm: (B, T, 7) 归一化的动作增量
        Returns:
            tokens: (B, T, 8, D) state 和 action token 拼接
        """
        B, T, _ = state_norm.shape
        
        # State → 4 tokens
        s_tok = self.state_proj(state_norm)                           # (B, T, 4*D)
        s_tok = s_tok.reshape(B, T, self.num_state_tokens, -1)       # (B, T, 4, D)
        s_tok = s_tok + self.state_modality_emb
        
        # Action → 4 tokens
        a_tok = self.action_proj(action_norm)                         # (B, T, 4*D)
        a_tok = a_tok.reshape(B, T, self.num_action_tokens, -1)      # (B, T, 4, D)
        a_tok = a_tok + self.action_modality_emb
        
        # 拼接: [state(4) | action(4)]
        return torch.cat([s_tok, a_tok], dim=2)                      # (B, T, 8, D)
    
    def decode(self, token_output):
        """
        Args:
            token_output: (B, T, 8, D) DiT 去噪后的输出
        Returns:
            pred_state:  (B, T, 7) 预测的绝对关节角
            pred_action: (B, T, 7) 预测的动作增量
        """
        B, T, K, D = token_output.shape
        
        # 拆分
        s_tok = token_output[:, :, :self.num_state_tokens]            # (B, T, 4, D)
        a_tok = token_output[:, :, self.num_state_tokens:]            # (B, T, 4, D)
        
        # 解码
        pred_state = self.state_output(
            s_tok.reshape(B, T, self.num_state_tokens * D)
        )                                                             # (B, T, 7)
        pred_action = self.action_output(
            a_tok.reshape(B, T, self.num_action_tokens * D)
        )                                                             # (B, T, 7)
        
        return pred_state, pred_action
参数量估算（hidden_dim=3072, 4+4 tokens）:
- state_proj: 7×256 + 256×12288 ≈ 3.2M
- action_proj: 同上 ≈ 3.2M
- state_output: 12288×256 + 256×7 ≈ 3.2M
- action_output: 同上 ≈ 3.2M
- modality_emb: 2 × 4×3072 ≈ 25K
- 总计约 12.8M（DiT 总参数的 ~0.6%，全量训练无压力）
2.2 S0Encoder
不变（和 v2 相同）。编码初始关节角为全局条件 token。
class S0Encoder(nn.Module):
    """
    初始关节角 → 条件 token，拼接到 text embedding 后面。
    所有 video/state/action token 通过 attention 都能看到。
    """
    def __init__(self, state_dim=7, hidden_dim=3072, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, num_tokens * hidden_dim),
        )
    
    def forward(self, s0_norm):
        """s0_norm: (B, 7) → (B, 4, D)"""
        B = s0_norm.shape[0]
        tokens = self.proj(s0_norm)
        tokens = tokens.reshape(B, self.num_tokens, -1)
        return tokens
2.3 StateActionAttentionBias（可选增强）
防止 8 个 state+action token 被 256 个 visual token 淹没。
class StateActionAttentionBias(nn.Module):
    """
    在每个 attention head 上给 state/action key 位置加可学习偏置。
    State 和 action 各有独立 bias，让模型分别学习关注度。
    """
    def __init__(self, num_heads):
        super().__init__()
        self.state_bias = nn.Parameter(torch.zeros(num_heads))
        self.action_bias = nn.Parameter(torch.zeros(num_heads))
    
    def apply(self, attn_logits, state_key_mask, action_key_mask):
        """
        attn_logits:     (B, H, N_q, N_k)
        state_key_mask:  (N_k,) bool
        action_key_mask: (N_k,) bool
        """
        s_bias = self.state_bias.view(1, -1, 1, 1)
        a_bias = self.action_bias.view(1, -1, 1, 1)
        attn_logits[:, :, :, state_key_mask] += s_bias
        attn_logits[:, :, :, action_key_mask] += a_bias
        return attn_logits

---
三、Token 序列构造
3.1 完整的序列构建函数
def build_full_sequence(text_tokens, s0_cond_tokens, visual_tokens, sa_tokens):
    """
    构建最终的 token 序列。
    
    Args:
        text_tokens:    (B, 226, D)     text embedding
        s0_cond_tokens: (B, 4, D)       S₀ 全局条件（clean）
        visual_tokens:  (B, 13, 256, D) 每帧 256 个 spatial patch
        sa_tokens:      (B, 13, 8, D)   每帧 [state(4) + action(4)] token
    
    Returns:
        sequence: (B, 3662, D)
        meta:     dict
    """
    B = text_tokens.shape[0]
    
    # 条件区
    condition = torch.cat([text_tokens, s0_cond_tokens], dim=1)     # (B, 230, D)
    
    # 视频+状态+动作区: 交替排列
    parts = []
    for t in range(13):
        parts.append(visual_tokens[:, t])    # (B, 256, D) Frame_t
        parts.append(sa_tokens[:, t])        # (B, 8, D)   [State_t + Action_t]
    video_sa = torch.cat(parts, dim=1)       # (B, 3432, D)
    
    sequence = torch.cat([condition, video_sa], dim=1)              # (B, 3662, D)
    
    # 元信息
    meta = {
        'text_range': (0, 226),
        's0_cond_range': (226, 230),
        'condition_range': (0, 230),
        'frame_indices': {},      # t → (start, end) visual patches
        'sa_indices': {},         # t → (start, end) state+action tokens
        'state_indices': {},      # t → (start, end) state tokens only
        'action_indices': {},     # t → (start, end) action tokens only
    }
    
    offset = 230
    for t in range(13):
        meta['frame_indices'][t] = (offset, offset + 256)
        offset += 256
        sa_start = offset
        meta['sa_indices'][t] = (sa_start, sa_start + 8)
        meta['state_indices'][t] = (sa_start, sa_start + 4)      # 前 4 个
        meta['action_indices'][t] = (sa_start + 4, sa_start + 8) # 后 4 个
        offset += 8
    
    return sequence, meta
3.2 从输出中提取各模态
def extract_outputs(output, meta):
    """
    从 DiT 输出中分离 visual、state、action 三个部分。
    
    Returns:
        visual_output: (B, 13, 256, D)
        sa_output:     (B, 13, 8, D)   [state(4) + action(4)]
    """
    visual_parts = []
    sa_parts = []
    
    for t in range(13):
        fs, fe = meta['frame_indices'][t]
        visual_parts.append(output[:, fs:fe])
        
        ss, se = meta['sa_indices'][t]
        sa_parts.append(output[:, ss:se])
    
    visual_output = torch.stack(visual_parts, dim=1)    # (B, 13, 256, D)
    sa_output = torch.stack(sa_parts, dim=1)            # (B, 13, 8, D)
    
    return visual_output, sa_output

---
四、Causal Mask 构造
和 v2 完全一致。State+Action 在同一时间步内互相可见（time_id 相同）。
4.1 Time ID 分配
def build_time_ids(num_text=226, num_s0=4, num_frames=13,
                   patches_per_frame=256, sa_per_frame=8):
    """
    条件区: time_id = -1
    Frame_t:         time_id = 2t     (偶数)
    [State+Action]_t: time_id = 2t+1  (奇数)
    """
    time_ids = []
    time_ids.extend([-1] * (num_text + num_s0))          # 条件区 230 个
    for t in range(num_frames):
        time_ids.extend([2 * t] * patches_per_frame)     # Frame_t: 256 个
        time_ids.extend([2 * t + 1] * sa_per_frame)      # [S,A]_t: 8 个
    return torch.tensor(time_ids)                         # (3662,)
4.2 Causal Mask
def build_causal_mask(time_ids):
    """和 v2 完全一致。time_j <= time_i 则允许 attend，条件区特殊处理。"""
    N = len(time_ids)
    is_cond = (time_ids == -1)
    
    mask = time_ids.unsqueeze(0) >= time_ids.unsqueeze(1)
    mask[:, is_cond] = True
    
    cond_idx = is_cond.nonzero(as_tuple=True)[0]
    non_cond_idx = (~is_cond).nonzero(as_tuple=True)[0]
    mask[cond_idx.unsqueeze(1), non_cond_idx.unsqueeze(0)] = False
    
    return mask
4.3 Chunked SDPA
和 v2 一致：每对 (Frame_t, [S,A]_t) = 264 token 为一个 chunk，13 次循环。 Chunk 内部 mask: Frame (前256) 不能看 [S,A] (后8)，但 [S,A] 能看 Frame。
def chunked_causal_sdpa(hidden, condition_kv, num_frames=13):
    """每个 chunk = Frame_t(256) + [S,A]_t(8) = 264 token"""
    outputs = []
    kv_accumulated = condition_kv   # (B, 230, D)
    
    for t in range(num_frames):
        chunk_start = 230 + t * 264
        chunk_q = hidden[:, chunk_start:chunk_start + 264]    # (B, 264, D)
        current_kv = torch.cat([kv_accumulated, chunk_q], dim=1)
        
        # 帧内 mask: 前256行不能看后8列
        len_kv = current_kv.shape[1]
        inner_mask = torch.ones(264, len_kv, dtype=torch.bool, device=hidden.device)
        inner_mask[:256, -8:] = False
        
        chunk_out = F.scaled_dot_product_attention(
            chunk_q, current_kv, current_kv, attn_mask=inner_mask
        )
        outputs.append(chunk_out)
        kv_accumulated = current_kv
    
    return torch.cat(outputs, dim=1)

---
五、位置编码适配
和 v2 一致。
def build_temporal_rope_ids(num_frames=13, patches_per_frame=256, sa_per_frame=8):
    """Frame_t → 2t, [S,A]_t → 2t+1, 共 26 个时间位置"""
    rope_ids = []
    for t in range(num_frames):
        rope_ids.extend([2 * t] * patches_per_frame)
        rope_ids.extend([2 * t + 1] * sa_per_frame)
    return torch.tensor(rope_ids)

def build_spatial_position_ids(num_frames=13, grid_h=16, grid_w=16, sa_per_frame=8):
    """Frame: (row, col), State+Action: 超出 grid 的虚拟位置"""
    spatial_ids = []
    for t in range(num_frames):
        for r in range(grid_h):
            for c in range(grid_w):
                spatial_ids.append((r, c))
        for k in range(sa_per_frame):
            spatial_ids.append((16 + k, 0))
    return spatial_ids

---
六、数据准备
6.1 GT 构造（同时准备 state 和 action）
def prepare_gt(state_seq_24, norm_stats):
    """
    从 BridgeV2 的 24 帧 state 序列同时构造 state GT 和 action GT。
    
    Args:
        state_seq_24: (B, 24, 7)   原始绝对关节角
        norm_stats:   dict         {'mean': (7,), 'std': (7,)}
    
    Returns:
        state_gt_13:  (B, 13, 7)   归一化绝对关节角
        action_gt_13: (B, 13, 7)   归一化动作增量（最后一步 pad 0）
        s0_norm:      (B, 7)       归一化初始关节角
    """
    mean, std = norm_stats['mean'], norm_stats['std']
    
    # 下采样 24 → 13
    indices = torch.linspace(0, 23, 13).long()
    state_13 = state_seq_24[:, indices]                         # (B, 13, 7)
    
    # State GT: 归一化绝对值
    state_gt_13 = (state_13 - mean) / std                      # (B, 13, 7)
    
    # Action GT: 归一化增量
    delta = state_13[:, 1:] - state_13[:, :-1]                 # (B, 12, 7)
    delta_norm = delta / std                                    # 用 std 归一化（不减 mean）
    action_gt_13 = F.pad(delta_norm, (0, 0, 0, 1), value=0)   # (B, 13, 7) 最后一步 pad 0
    
    # S₀
    s0_norm = state_gt_13[:, 0]                                # (B, 7)
    
    return state_gt_13, action_gt_13, s0_norm
6.2 预计算归一化统计量
def compute_norm_stats(dataset):
    """对训练集所有帧的绝对关节角计算 mean 和 std。运行一次，保存复用。"""
    all_states = []
    for trajectory in dataset:
        states = trajectory['observation/state']
        all_states.append(states)
    all_states = np.concatenate(all_states, axis=0)
    
    stats = {
        'mean': torch.tensor(all_states.mean(axis=0), dtype=torch.float32),
        'std': torch.tensor(all_states.std(axis=0), dtype=torch.float32).clamp(min=1e-6),
    }
    torch.save(stats, 'state_norm_stats.pt')
    return stats

---
七、Loss 设计
v3 的核心优势：三项 loss 互相约束。
def compute_sa_loss(sa_output, state_tokenizer, state_gt_13, action_gt_13,
                    lambda_s=1.0, lambda_a=1.0, lambda_c=0.5):
    """
    计算 State + Action + Consistency 三项 loss。
    
    Args:
        sa_output:     (B, 13, 8, D) DiT 输出的 state+action token
        state_tokenizer: StateActionTokenizer 实例
        state_gt_13:   (B, 13, 7)    GT 绝对关节角
        action_gt_13:  (B, 13, 7)    GT 增量
    
    Returns:
        loss_dict: dict 各项 loss
    """
    # 解码
    pred_state, pred_action = state_tokenizer.decode(sa_output)
    # pred_state:  (B, 13, 7) 预测的绝对关节角
    # pred_action: (B, 13, 7) 预测的动作增量
    
    # L_state: 绝对位置准确
    L_state = F.mse_loss(pred_state, state_gt_13)
    
    # L_action: 动作增量准确
    L_action = F.mse_loss(pred_action, action_gt_13)
    
    # L_consistency: state 和 action 互相一致
    # 从 state 推出的隐含 delta 应该等于直接预测的 action
    implied_delta = pred_state[:, 1:] - pred_state[:, :-1]       # (B, 12, 7)
    predicted_delta = pred_action[:, :-1]                         # (B, 12, 7)
    L_consistency = F.mse_loss(implied_delta, predicted_delta)
    
    # 加权总和
    L_sa = lambda_s * L_state + lambda_a * L_action + lambda_c * L_consistency
    
    return {
        'L_state': L_state,
        'L_action': L_action,
        'L_consistency': L_consistency,
        'L_sa': L_sa,
    }
三项 loss 的职责：
暂时无法在飞书文档外展示此内容

---
八、三阶段训练实现
8.1 阶段 1: [State,Action] = Clean GT, 训视频去噪
def train_stage1(dit, sa_tokenizer, s0_encoder, batch, scheduler, optimizer):
    """
    DiT 在已知精确状态轨迹和动作序列的条件下学习生成高质量视频。
    State+Action tokens 不加噪声，作为精确条件参与 self-attention。
    """
    # 准备数据
    visual_tokens = patchify(vae.encode(batch['video']))                   # (B, 13, 256, D)
    state_gt, action_gt, s0_norm = prepare_gt(batch['state_seq'], norm_stats)
    text_tokens = text_encoder(batch['text'])                              # (B, 226, D)
    s0_cond_tokens = s0_encoder(s0_norm)                                   # (B, 4, D)
    
    # State+Action: clean GT（不加噪声）
    clean_sa_tokens = sa_tokenizer.encode(state_gt, action_gt)             # (B, 13, 8, D)
    
    # Visual: 加噪声
    timestep = torch.randint(0, 1000, (B,), device=device)
    noise = torch.randn_like(visual_tokens)
    noisy_visual = scheduler.add_noise(visual_tokens, noise, timestep)
    
    # 构建序列 + DiT 前向
    sequence, meta = build_full_sequence(text_tokens, s0_cond_tokens, noisy_visual, clean_sa_tokens)
    output = dit(sequence, timestep, causal_mask)
    
    # Loss: 只算视频
    visual_output, _ = extract_outputs(output, meta)
    L_video = F.mse_loss(visual_output, noise)
    
    optimizer.zero_grad()
    L_video.backward()
    optimizer.step()
    
    return {'L_video': L_video.item()}
训练配置:
暂时无法在飞书文档外展示此内容
8.2 阶段 2: Visual = Clean GT, 训 State+Action 去噪
def train_stage2(dit, sa_tokenizer, s0_encoder, batch, scheduler, optimizer_sa):
    """
    State+Action 在已知视频内容条件下学习去噪。DiT 全部冻结。
    """
    # Clean 数据（无梯度）
    with torch.no_grad():
        visual_tokens = patchify(vae.encode(batch['video']))               # clean
        text_tokens = text_encoder(batch['text'])
        s0_cond_tokens = s0_encoder(s0_norm)
    
    # State+Action: 加噪声
    state_gt, action_gt, _ = prepare_gt(batch['state_seq'], norm_stats)
    clean_sa_tokens = sa_tokenizer.encode(state_gt, action_gt)
    noise_sa = torch.randn_like(clean_sa_tokens)
    timestep = torch.randint(0, 1000, (B,), device=device)
    noisy_sa = scheduler.add_noise(clean_sa_tokens, noise_sa, timestep)
    
    # 构建序列 + DiT 前向
    sequence, meta = build_full_sequence(text_tokens, s0_cond_tokens, visual_tokens, noisy_sa)
    output = dit(sequence, timestep, causal_mask)
    
    # Loss: state + action + consistency
    _, sa_output = extract_outputs(output, meta)
    loss_dict = compute_sa_loss(sa_output, sa_tokenizer, state_gt, action_gt,
                                lambda_s=1.0, lambda_a=1.0, lambda_c=0.5)
    
    optimizer_sa.zero_grad()
    loss_dict['L_sa'].backward()
    optimizer_sa.step()
    
    return {k: v.item() for k, v in loss_dict.items()}
训练配置:
暂时无法在飞书文档外展示此内容
8.3 阶段 3: 联合去噪
def train_stage3(dit, sa_tokenizer, s0_encoder, batch, scheduler, optimizer_joint):
    """Video 和 State+Action 联合去噪，三方互相约束。"""
    # 准备 clean 数据
    visual_tokens = patchify(vae.encode(batch['video']))
    state_gt, action_gt, s0_norm = prepare_gt(batch['state_seq'], norm_stats)
    clean_sa_tokens = sa_tokenizer.encode(state_gt, action_gt)
    text_tokens = text_encoder(batch['text'])
    s0_cond_tokens = s0_encoder(s0_norm)
    
    # 两个模态同一个 timestep 加噪
    timestep = torch.randint(0, 1000, (B,), device=device)
    noise_visual = torch.randn_like(visual_tokens)
    noise_sa = torch.randn_like(clean_sa_tokens)
    noisy_visual = scheduler.add_noise(visual_tokens, noise_visual, timestep)
    noisy_sa = scheduler.add_noise(clean_sa_tokens, noise_sa, timestep)
    
    # 构建序列 + DiT 前向
    sequence, meta = build_full_sequence(text_tokens, s0_cond_tokens, noisy_visual, noisy_sa)
    output = dit(sequence, timestep, causal_mask)
    
    # 提取输出
    visual_output, sa_output = extract_outputs(output, meta)
    
    # 联合 Loss
    L_video = F.mse_loss(visual_output, noise_visual)
    loss_dict = compute_sa_loss(sa_output, sa_tokenizer, state_gt, action_gt,
                                lambda_s=1.0, lambda_a=1.0, lambda_c=0.5)
    
    lambda_sa = 0.1
    L_total = L_video + lambda_sa * loss_dict['L_sa']
    
    optimizer_joint.zero_grad()
    L_total.backward()
    optimizer_joint.step()
    
    return {
        'L_video': L_video.item(),
        'L_state': loss_dict['L_state'].item(),
        'L_action': loss_dict['L_action'].item(),
        'L_consistency': loss_dict['L_consistency'].item(),
        'L_total': L_total.item(),
    }
训练配置:
暂时无法在飞书文档外展示此内容

---
九、推理流程
@torch.no_grad()
def inference(dit, sa_tokenizer, s0_encoder, text, first_frame_rgb, s0,
              scheduler, norm_stats, num_steps=50):
    """
    从纯噪声同时生成 49 帧视频 + 状态轨迹 + 动作序列。
    
    Args:
        text: str                         任务指令
        first_frame_rgb: (1, 3, 256, 256) 首帧 RGB
        s0: (1, 7)                        初始关节角
    
    Returns:
        video:          (49, 3, 256, 256) 生成视频
        state_traj_24:  (24, 7)           绝对关节角轨迹
        action_delta_23:(23, 7)           动作增量序列
    """
    # 编码条件
    text_tokens = text_encoder(text)
    s0_norm = (s0 - norm_stats['mean']) / norm_stats['std']
    s0_cond_tokens = s0_encoder(s0_norm)
    condition_latent = prepare_i2v_condition(first_frame_rgb)
    
    # 初始化: 从纯噪声开始
    noisy_visual = torch.randn(1, 13, 256, D, device=device)
    noisy_sa = torch.randn(1, 13, 8, D, device=device)
    
    # Causal mask（构建一次）
    time_ids = build_time_ids()
    causal_mask = build_causal_mask(time_ids).to(device)
    
    # 去噪循环
    for t in scheduler.timesteps:
        sequence, meta = build_full_sequence(
            text_tokens, s0_cond_tokens, noisy_visual, noisy_sa
        )
        output = dit(sequence, t, causal_mask, condition_latent)
        visual_pred, sa_pred = extract_outputs(output, meta)
        
        noisy_visual = scheduler.step(visual_pred, t, noisy_visual)
        noisy_sa = scheduler.step(sa_pred, t, noisy_sa)
    
    # === 解码视频 ===
    video = vae.decode(unpatchify(noisy_visual))                        # (1, 49, 3, H, W)
    
    # === 解码 state + action ===
    pred_state_13, pred_action_13 = sa_tokenizer.decode(noisy_sa)       # 各 (1, 13, 7)
    
    # 反归一化
    mean, std = norm_stats['mean'], norm_stats['std']
    pred_state_13 = pred_state_13 * std + mean                          # 绝对关节角
    pred_action_13 = pred_action_13 * std                               # 增量（只乘 std）
    
    # 上采样 13 → 24 帧
    pred_state_24 = F.interpolate(
        pred_state_13.permute(0, 2, 1), size=24,
        mode='linear', align_corners=True
    ).permute(0, 2, 1)                                                  # (1, 24, 7)
    
    # Action delta: 优先用 state 差分（比直接用 action 输出更稳定）
    # 也可以对 action_13 做上采样，取决于哪个精度更高
    action_delta_23 = pred_state_24[:, 1:] - pred_state_24[:, :-1]     # (1, 23, 7)
    
    return (
        video.squeeze(0),                 # (49, 3, H, W)
        pred_state_24.squeeze(0),         # (24, 7) 绝对轨迹
        action_delta_23.squeeze(0),       # (23, 7) 增量（发给控制器）
    )

---
十、验证检查点
10.1 阶段 1 完成后
暂时无法在飞书文档外展示此内容
10.2 阶段 2 完成后
暂时无法在飞书文档外展示此内容
10.3 阶段 3 完成后
暂时无法在飞书文档外展示此内容
10.4 消融实验清单
暂时无法在飞书文档外展示此内容

---
十一、执行时间线
准备工作
├─ 打印 CogVideoX 具体数字（hidden_dim, heads, RoPE, patchify）   半天
├─ 预计算 state 归一化统计量                                       1 小时
└─ 实现 StateActionTokenizer + S0Encoder + Bias                    1 天

核心改动
├─ 修改 token 序列构造（build_full_sequence）                       1 天
├─ 适配 causal mask（time_id 26 位置 + mask + verify）             1 天
├─ 适配 chunked SDPA（264 token/chunk + 帧内非对称 mask）          1 天
├─ 适配 RoPE（13 → 26 时间位置）                                   半天
└─ 修改输出逻辑（unpatchify 前剥离 state+action token）            半天

训练
├─ 阶段 1: clean [S,A] + 训视频           3000-5000 步             2-3 天
├─ 阶段 1 验证（视频质量 + token 影响力）                           半天
├─ 阶段 2: clean visual + 训 [S,A] 去噪   1000-2000 步             1-2 天
├─ 阶段 2 验证（MSE + consistency + FK）                            半天
├─ 阶段 3: 联合去噪                        2000-3000 步             2-3 天
└─ 阶段 3 验证 + 端到端推理                                         1 天

消融实验（论文用）
├─ State+Action vs 纯 State vs 纯 Action                           按需
├─ 有无 L_consistency                                               3-4 天
├─ Token 数量消融                                                   按需
└─ 三阶段 vs 直接联合                                               3-4 天
────────────────────────────────────────────────────
核心开发+训练总计                                                    约 12-16 天

---
十二、风险与缓解
暂时无法在飞书文档外展示此内容

