# Kronos 模型架构完整分析

本文档详细分析 Kronos 项目中的两个核心模型：**KronosTokenizer** 和 **Kronos Predictor**，以及它们的微调训练流程。

---

## 📋 目录

1. [整体架构概览](#整体架构概览)
2. [KronosTokenizer 详细架构](#kronostokenizer-详细架构)
3. [Kronos Predictor 详细架构](#kronos-predictor-详细架构)
4. [核心模块解析](#核心模块解析)
5. [训练流程分析](#训练流程分析)
6. [数据流与前向传播](#数据流与前向传播)
7. [损失函数设计](#损失函数设计)
8. [推理流程](#推理流程)

---

## 🏗️ 整体架构概览

Kronos 采用**两阶段架构**，将时间序列预测分解为：

```
原始时间序列 → [Tokenizer编码] → 离散Token序列 → [Predictor预测] → 未来Token序列 → [Tokenizer解码] → 预测序列
```

### 设计理念

1. **Tokenization**: 将连续的金融时间序列压缩为离散的token表示（类似NLP中的词向量化）
2. **Sequence Modeling**: 使用Transformer预测token序列（类似语言模型）
3. **Hierarchical Quantization**: 采用两级量化（s1_bits + s2_bits）提高表达能力

### 文件组织结构

```
model/
├── kronos.py           # 主模型定义（KronosTokenizer, Kronos, KronosPredictor）
└── module.py           # 基础模块（Attention, FFN, Quantizer等）

finetune/
├── train_tokenizer.py  # Tokenizer微调（GPU分布式）
├── train_tokenizer_cpu.py  # Tokenizer微调（CPU单机）
├── train_predictor.py  # Predictor微调（GPU分布式）
├── train_predictor_cpu.py  # Predictor微调（CPU单机）
├── config.py           # 配置参数
└── dataset.py          # 数据集加载器
```

---

## 🔧 KronosTokenizer 详细架构

### 1. 整体结构

**KronosTokenizer** 是一个基于 **VQVAE (Vector Quantized Variational AutoEncoder)** 的编码器-解码器架构，使用 **Binary Spherical Quantization (BSQ)** 进行离散化。

```python
class KronosTokenizer(nn.Module):
    """
    输入: (batch_size, seq_len, d_in=6)  # 6特征: open, high, low, close, vol, amt
    输出: (batch_size, seq_len, d_in=6)  # 重建后的序列
    """
```

### 2. 架构组件

#### 2.1 输入嵌入层
```python
self.embed = nn.Linear(d_in, d_model)  # d_in=6 → d_model (e.g., 256)
```

#### 2.2 编码器 (Encoder)
```python
self.encoder = nn.ModuleList([
    TransformerBlock(d_model, n_heads, ff_dim, dropout_params)
    for _ in range(n_enc_layers - 1)  # 例如: 4层Transformer
])
```

**TransformerBlock 结构**:
- RMSNorm (Pre-Norm)
- MultiHeadAttentionWithRoPE (带旋转位置编码的自注意力)
- RMSNorm (Pre-Norm)
- FeedForward (SwiGLU激活函数)

#### 2.3 量化模块 (BSQuantizer)

```python
self.quant_embed = nn.Linear(d_model, codebook_dim)  # codebook_dim = s1_bits + s2_bits
self.tokenizer = BSQuantizer(s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size)
```

**量化过程**:
1. **L2归一化**: 将输入向量归一化到单位球面
2. **二值量化**: 每个维度量化为 {-1, +1}
3. **两级分层**:
   - **s1_bits**: 前半部分 (粗粒度表示)
   - **s2_bits**: 后半部分 (细粒度表示)
4. **熵正则化**: 鼓励码本的均匀使用

**BSQuantizer 损失函数**:
```python
commit_loss = beta * ||z_quantized - z||^2  # 承诺损失
entropy_penalty = gamma0 * H_per_sample - gamma * H_codebook  # 熵正则
bsq_loss = commit_loss + zeta * entropy_penalty
```

#### 2.4 解码器 (Decoder)

**两路并行解码**:

```python
# 路径1: 仅使用 s1_bits (粗粒度重建)
self.post_quant_embed_pre = nn.Linear(s1_bits, d_model)
# 解码器Transformer层
z_pre = head(decoder_layers(post_quant_embed_pre(quantized[:,:,:s1_bits])))

# 路径2: 使用完整 codebook (精细重建)
self.post_quant_embed = nn.Linear(codebook_dim, d_model)
z = head(decoder_layers(post_quant_embed(quantized)))
```

```python
self.decoder = nn.ModuleList([
    TransformerBlock(d_model, n_heads, ff_dim, dropout_params)
    for _ in range(n_dec_layers - 1)  # 例如: 4层Transformer
])
self.head = nn.Linear(d_model, d_in)  # 投影回原始特征空间
```

### 3. 前向传播流程

```python
def forward(self, x):
    # x: [B, T, 6]
    
    # 1. 编码
    z = self.embed(x)  # [B, T, d_model]
    for layer in self.encoder:
        z = layer(z)
    
    # 2. 量化准备
    z = self.quant_embed(z)  # [B, T, codebook_dim]
    
    # 3. BSQ量化
    bsq_loss, quantized, z_indices = self.tokenizer(z)
    # quantized: [B, T, codebook_dim]
    # z_indices: 离散token索引
    
    # 4. 双路解码
    # 路径1: s1解码
    quantized_pre = quantized[:, :, :s1_bits]
    z_pre = self.post_quant_embed_pre(quantized_pre)
    for layer in self.decoder:
        z_pre = layer(z_pre)
    z_pre = self.head(z_pre)  # [B, T, 6]
    
    # 路径2: 完整解码
    z = self.post_quant_embed(quantized)
    for layer in self.decoder:
        z = layer(z)
    z = self.head(z)  # [B, T, 6]
    
    return (z_pre, z), bsq_loss, quantized, z_indices
```

### 4. 编码与解码接口

```python
def encode(self, x, half=False):
    """将时间序列编码为token ID"""
    # 返回: z_indices (如果half=True则返回[s1_indices, s2_indices])

def decode(self, x, half=False):
    """将token ID解码回时间序列"""
    quantized = self.indices_to_bits(x, half)  # ID转比特表示
    z = self.post_quant_embed(quantized)
    for layer in self.decoder:
        z = layer(z)
    return self.head(z)
```

### 5. 参数配置示例

```python
KronosTokenizer(
    d_in=6,              # 输入特征维度
    d_model=256,         # 模型隐藏层维度
    n_heads=8,           # 注意力头数
    ff_dim=1024,         # FFN中间层维度
    n_enc_layers=4,      # 编码器层数
    n_dec_layers=4,      # 解码器层数
    s1_bits=10,          # 粗粒度量化位数 (词汇表大小: 2^10=1024)
    s2_bits=10,          # 细粒度量化位数
    beta=0.25,           # 承诺损失权重
    gamma0=1.0,          # 样本熵权重
    gamma=1.0,           # 码本熵权重
    zeta=1.0,            # 总熵正则权重
    group_size=10        # 分组大小
)
```

---

## 🤖 Kronos Predictor 详细架构

### 1. 整体结构

**Kronos Predictor** 是一个 **Transformer语言模型**，用于自回归预测token序列。

```python
class Kronos(nn.Module):
    """
    输入: 
        - s1_ids: [batch_size, seq_len] (token序列 - 粗粒度)
        - s2_ids: [batch_size, seq_len] (token序列 - 细粒度)
        - stamp: [batch_size, seq_len, 5] (时间特征)
    输出:
        - s1_logits: [batch_size, seq_len, 2^s1_bits] (s1预测概率)
        - s2_logits: [batch_size, seq_len, 2^s2_bits] (s2预测概率|s1)
    """
```

### 2. 架构组件

#### 2.1 嵌入层

```python
# 层级嵌入: 融合s1和s2的嵌入
self.embedding = HierarchicalEmbedding(s1_bits, s2_bits, d_model)
```

**HierarchicalEmbedding 内部结构**:
```python
class HierarchicalEmbedding(nn.Module):
    def __init__(self, s1_bits, s2_bits, d_model):
        self.emb_s1 = nn.Embedding(2**s1_bits, d_model)  # s1词汇表
        self.emb_s2 = nn.Embedding(2**s2_bits, d_model)  # s2词汇表
        self.fusion_proj = nn.Linear(d_model*2, d_model) # 融合投影
    
    def forward(self, token_ids):
        s1_ids, s2_ids = token_ids
        s1_emb = self.emb_s1(s1_ids) * sqrt(d_model)
        s2_emb = self.emb_s2(s2_ids) * sqrt(d_model)
        return self.fusion_proj(cat([s1_emb, s2_emb]))  # [B, T, d_model]
```

#### 2.2 时间嵌入

```python
self.time_emb = TemporalEmbedding(d_model, learn_te)
```

**时间特征嵌入**:
```python
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, learn_pe):
        self.minute_embed = Embedding(60, d_model)
        self.hour_embed = Embedding(24, d_model)
        self.weekday_embed = Embedding(7, d_model)
        self.day_embed = Embedding(32, d_model)
        self.month_embed = Embedding(13, d_model)
    
    def forward(self, x):
        # x: [B, T, 5] -> [minute, hour, weekday, day, month]
        return (minute_emb + hour_emb + weekday_emb + day_emb + month_emb)
```

#### 2.3 Transformer主干

```python
self.transformer = nn.ModuleList([
    TransformerBlock(d_model, n_heads, ff_dim, dropout_params)
    for _ in range(n_layers)  # 例如: 12层
])
self.norm = RMSNorm(d_model)
```

#### 2.4 依赖感知层 (Dependency-Aware Layer)

用于建模 s2 对 s1 的**条件依赖**关系：

```python
self.dep_layer = DependencyAwareLayer(d_model)
```

**内部结构**:
```python
class DependencyAwareLayer(nn.Module):
    def __init__(self, d_model, n_heads=4):
        self.cross_attn = MultiHeadCrossAttentionWithRoPE(d_model, n_heads)
        self.norm = RMSNorm(d_model)
    
    def forward(self, hidden_states, sibling_embed, key_padding_mask=None):
        # hidden_states: 来自Transformer的上下文表示
        # sibling_embed: s1的嵌入表示
        attn_out = self.cross_attn(
            query=sibling_embed,      # 用s1作为query
            key=hidden_states,         # 用上下文作为key
            value=hidden_states        # 用上下文作为value
        )
        return self.norm(hidden_states + attn_out)
```

#### 2.5 双头预测 (Dual Head)

```python
self.head = DualHead(s1_bits, s2_bits, d_model)
```

**DualHead 结构**:
```python
class DualHead(nn.Module):
    def __init__(self, s1_bits, s2_bits, d_model):
        self.vocab_s1 = 2 ** s1_bits
        self.vocab_s2 = 2 ** s2_bits
        self.proj_s1 = nn.Linear(d_model, self.vocab_s1)  # s1预测头
        self.proj_s2 = nn.Linear(d_model, self.vocab_s2)  # s2预测头
    
    def forward(self, x):
        return self.proj_s1(x)  # 预测s1
    
    def cond_forward(self, x2):
        return self.proj_s2(x2)  # 预测s2（条件于s1）
```

### 3. 前向传播流程

```python
def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None, 
            use_teacher_forcing=False, s1_targets=None):
    # 1. Token嵌入 + 时间嵌入
    x = self.embedding([s1_ids, s2_ids])  # [B, T, d_model]
    if stamp is not None:
        time_embedding = self.time_emb(stamp)
        x = x + time_embedding
    x = self.token_drop(x)
    
    # 2. Transformer编码
    for layer in self.transformer:
        x = layer(x, key_padding_mask=padding_mask)  # 因果注意力
    x = self.norm(x)  # [B, T, d_model]
    
    # 3. 预测 s1
    s1_logits = self.head(x)  # [B, T, vocab_s1]
    
    # 4. 获取s1的嵌入（用于条件预测s2）
    if use_teacher_forcing:
        sibling_embed = self.embedding.emb_s1(s1_targets)  # 训练时：真实s1
    else:
        # 推理时：从s1_logits采样
        s1_probs = F.softmax(s1_logits.detach(), dim=-1)
        sample_s1_ids = torch.multinomial(s1_probs, 1)
        sibling_embed = self.embedding.emb_s1(sample_s1_ids)
    
    # 5. 依赖感知层：条件于s1预测s2
    x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)
    s2_logits = self.head.cond_forward(x2)  # [B, T, vocab_s2]
    
    return s1_logits, s2_logits
```

### 4. 解码接口（推理用）

```python
def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
    """只预测s1，返回s1_logits和context"""
    x = self.embedding([s1_ids, s2_ids])
    if stamp is not None:
        x = x + self.time_emb(stamp)
    x = self.token_drop(x)
    
    for layer in self.transformer:
        x = layer(x, key_padding_mask=padding_mask)
    x = self.norm(x)
    
    s1_logits = self.head(x)
    return s1_logits, x  # 返回context供s2解码使用

def decode_s2(self, context, s1_ids, padding_mask=None):
    """基于context和采样的s1预测s2"""
    sibling_embed = self.embedding.emb_s1(s1_ids)
    x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
    return self.head.cond_forward(x2)
```

### 5. 参数配置示例

```python
Kronos(
    s1_bits=10,              # s1词汇表: 2^10=1024
    s2_bits=10,              # s2词汇表: 2^10=1024
    n_layers=12,             # Transformer层数
    d_model=512,             # 隐藏层维度
    n_heads=8,               # 注意力头数
    ff_dim=2048,             # FFN维度
    ffn_dropout_p=0.1,
    attn_dropout_p=0.1,
    resid_dropout_p=0.1,
    token_dropout_p=0.1,
    learn_te=True            # 是否学习时间嵌入
)
```

---

## 🧩 核心模块解析

### 1. TransformerBlock

采用 **Pre-Norm** 架构（与原始Transformer的Post-Norm不同）：

```python
class TransformerBlock(nn.Module):
    def forward(self, x, key_padding_mask=None):
        # 自注意力块
        residual = x
        x = self.norm1(x)  # Pre-Norm
        attn_out = self.self_attn(x, key_padding_mask)
        x = residual + attn_out
        
        # FFN块
        residual = x
        x = self.norm2(x)  # Pre-Norm
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        return x
```

### 2. MultiHeadAttentionWithRoPE

**旋转位置编码 (RoPE)** 替代传统的位置编码：

```python
class MultiHeadAttentionWithRoPE(nn.Module):
    def forward(self, x, key_padding_mask=None):
        # 1. Q, K, V投影
        q = self.q_proj(x)  # [B, T, d_model] -> [B, n_heads, T, head_dim]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. 应用RoPE
        q, k = self.rotary(q, k)
        
        # 3. 因果自注意力（下三角mask）
        attn_output = scaled_dot_product_attention(
            q, k, v,
            attn_mask=key_padding_mask,
            is_causal=True  # 防止看到未来信息
        )
        
        # 4. 输出投影
        return self.resid_dropout(self.out_proj(attn_output))
```

**RoPE 机制**:
```python
class RotaryPositionalEmbedding(nn.Module):
    def forward(self, q, k):
        # 计算旋转矩阵
        t = torch.arange(seq_len)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        
        # 旋转q和k
        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot
```

### 3. FeedForward (SwiGLU)

采用 **SwiGLU** 激活函数（性能优于ReLU/GELU）：

```python
class FeedForward(nn.Module):
    def forward(self, x):
        # SwiGLU: SiLU(W1(x)) ⊙ W3(x)
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

### 4. BSQuantizer

**Binary Spherical Quantization** 的核心优势：
- **球面约束**: 量化在单位超球面上进行，保持几何结构
- **二值化**: 每个维度只有两个值 {-1, +1}，存储高效
- **熵正则化**: 平衡码本利用率和表达能力

```python
class BinarySphericalQuantizer(nn.Module):
    def forward(self, z):
        # 1. 量化到 {-1, +1}
        zq = sign(z) with straight-through estimator
        
        # 2. 计算熵损失
        persample_entropy = H(z的每个样本的分布)  # 希望高：表达能力强
        codebook_entropy = H(整个批次的码本使用分布)  # 希望高：码本均匀使用
        
        # 3. 承诺损失
        commit_loss = ||zq - z||^2  # 使原始编码接近量化后的值
        
        total_loss = commit_loss + entropy_regularization
        return zq, total_loss, metrics
```

---

## 🎯 训练流程分析

### 1. Tokenizer微调流程

#### 训练目标
学习将原始时间序列编码为紧凑的离散表示，并能准确重建。

#### 损失函数

```python
def train_step(batch_x):
    # 前向传播
    (z_pre, z), bsq_loss, _, _ = model(batch_x)
    
    # 重建损失
    recon_loss_pre = F.mse_loss(z_pre, batch_x)  # s1重建损失
    recon_loss_all = F.mse_loss(z, batch_x)      # 完整重建损失
    recon_loss = recon_loss_pre + recon_loss_all
    
    # 总损失
    loss = (recon_loss + bsq_loss) / 2
    
    return loss
```

**损失组成**:
1. **recon_loss_pre**: 仅用s1_bits重建的MSE损失
2. **recon_loss_all**: 用完整codebook重建的MSE损失
3. **bsq_loss**: 量化损失（commit loss + entropy penalty）

#### 训练配置

```python
# 优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,  # tokenizer_learning_rate
    weight_decay=0.1
)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=2e-4,
    steps_per_epoch=len(train_loader),
    epochs=30,
    pct_start=0.03,  # 预热3%
    div_factor=10    # 初始lr = max_lr/10
)

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
```

#### 数据流

```python
# 输入: 原始时间序列
batch_x: [batch_size, seq_len=101, features=6]
# seq_len = lookback_window(90) + predict_window(10) + 1

# 数据归一化（实例级）
x_mean = mean(batch_x, axis=0)
x_std = std(batch_x, axis=0)
batch_x = (batch_x - x_mean) / (x_std + 1e-5)
batch_x = clip(batch_x, -5, 5)

# 训练
for epoch in epochs:
    for batch_x, _ in train_loader:
        loss = train_step(batch_x)
        loss.backward()
        optimizer.step()
```

### 2. Predictor微调流程

#### 训练目标
学习预测下一个token，类似语言模型的next-token prediction。

#### 数据准备

```python
# 1. 使用训练好的tokenizer编码
with torch.no_grad():
    token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
    # token_seq_0: [B, T] s1 token IDs
    # token_seq_1: [B, T] s2 token IDs

# 2. 准备输入和目标（teacher forcing）
token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]  # 去掉最后一个token
token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]   # 去掉第一个token
```

#### 损失函数

```python
def train_step(batch_x, batch_x_stamp):
    # 1. Tokenize
    token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
    
    # 2. 前向传播
    s1_logits, s2_logits = model(
        token_in[0], token_in[1], 
        batch_x_stamp[:, :-1, :]
    )
    # s1_logits: [B, T-1, vocab_s1]
    # s2_logits: [B, T-1, vocab_s2]
    
    # 3. 计算损失
    loss, s1_loss, s2_loss = model.head.compute_loss(
        s1_logits, s2_logits,
        token_out[0], token_out[1]
    )
    
    return loss
```

**DualHead.compute_loss 实现**:
```python
def compute_loss(self, s1_logits, s2_logits, s1_targets, s2_targets):
    # 交叉熵损失
    ce_s1 = F.cross_entropy(
        s1_logits.reshape(-1, vocab_s1), 
        s1_targets.reshape(-1)
    )
    ce_s2 = F.cross_entropy(
        s2_logits.reshape(-1, vocab_s2), 
        s2_targets.reshape(-1)
    )
    
    # 平均
    ce_loss = (ce_s1 + ce_s2) / 2
    return ce_loss, ce_s1, ce_s2
```

#### 训练配置

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=4e-5,  # predictor_learning_rate (比tokenizer小)
    betas=(0.9, 0.95),
    weight_decay=0.1
)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=4e-5,
    steps_per_epoch=len(train_loader),
    epochs=30,
    pct_start=0.03,
    div_factor=10
)

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
```

### 3. 训练技巧

#### 3.1 梯度累积
```python
for j in range(accumulation_steps):
    # 分割batch
    sub_batch = batch[j*sub_size:(j+1)*sub_size]
    loss = forward(sub_batch) / accumulation_steps
    loss.backward()  # 累积梯度

optimizer.step()
optimizer.zero_grad()
```

#### 3.2 分布式训练 (DDP)
```python
# 初始化
dist.init_process_group(backend='nccl')
model = DDP(model, device_ids=[local_rank])

# 训练
for batch in train_loader:
    loss = train_step(batch)
    loss.backward()
    optimizer.step()

# 验证时同步损失
dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
```

#### 3.3 Epoch级随机种子
```python
# 保证每个epoch的采样不同但可复现
train_dataset.set_epoch_seed(epoch_idx * 10000 + rank)
```

---

## 🔄 数据流与前向传播

### 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        原始数据                                  │
│  DataFrame: [open, high, low, close, volume, amount]           │
│  TimeStamps: [datetime index]                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    数据预处理 (Dataset)                          │
│  1. 提取滑动窗口 (lookback_window + predict_window + 1)         │
│  2. 生成时间特征 [minute, hour, weekday, day, month]            │
│  3. 实例级归一化: (x - mean) / std                               │
│  4. 裁剪: clip(x, -5, 5)                                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────┐            ┌──────────────────┐
│   batch_x       │            │  batch_x_stamp   │
│  [B, 101, 6]    │            │   [B, 101, 5]    │
└────────┬────────┘            └────────┬─────────┘
         │                               │
         │  (Training Tokenizer)         │
         ▼                               │
┌─────────────────────────────────┐     │
│     KronosTokenizer.forward     │     │
│  1. Embed: [B,101,6]->[B,101,256]│     │
│  2. Encoder Transformers         │     │
│  3. BSQ Quantization             │     │
│  4. Decoder Transformers         │     │
│  5. Head: [B,101,256]->[B,101,6] │     │
│  Loss: recon + bsq               │     │
└─────────────────────────────────┘     │
         │                               │
         │  (After Tokenizer Trained)    │
         ▼                               │
┌─────────────────────────────────┐     │
│    tokenizer.encode(batch_x)    │     │
│  返回: [token_seq_0, token_seq_1]│     │
│        [B, 101] each             │     │
└────────┬────────────────────────┘     │
         │                               │
         │  (Training Predictor)         │
         ▼                               ▼
┌─────────────────────────────────────────────┐
│         Kronos.forward                      │
│  Inputs:                                    │
│    - token_in: [s1[:, :-1], s2[:, :-1]]    │
│    - stamp: batch_x_stamp[:, :-1, :]       │
│  1. HierarchicalEmbedding([s1, s2])        │
│  2. + TemporalEmbedding(stamp)             │
│  3. Transformer Layers (causal attention)  │
│  4. Head(x) -> s1_logits                   │
│  5. DependencyAwareLayer -> s2_logits      │
│  Loss: CE(s1_logits, s1_targets) +         │
│        CE(s2_logits, s2_targets)           │
└─────────────────────────────────────────────┘
         │
         │  (Inference: Autoregressive)
         ▼
┌─────────────────────────────────────────────┐
│   auto_regressive_inference                 │
│  for t in range(pred_len):                  │
│    1. s1_logits, ctx = model.decode_s1()    │
│    2. sample s1 from s1_logits              │
│    3. s2_logits = model.decode_s2(ctx, s1)  │
│    4. sample s2 from s2_logits              │
│    5. append [s1, s2] to sequence           │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│    tokenizer.decode([s1_seq, s2_seq])       │
│  返回: 预测的时间序列 [B, pred_len, 6]       │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│            后处理                            │
│  1. 反归一化: pred * std + mean              │
│  2. 构造DataFrame with timestamps            │
└─────────────────────────────────────────────┘
```

### 数据维度变化追踪

**Tokenizer训练阶段**:
```
输入:  [B=50, T=101, F=6]
    ↓ embed
       [B=50, T=101, D=256]
    ↓ encoder (4 layers)
       [B=50, T=101, D=256]
    ↓ quant_embed
       [B=50, T=101, C=20]  # codebook_dim = s1_bits(10) + s2_bits(10)
    ↓ BSQ quantization
       quantized: [B=50, T=101, C=20]
       indices: [B=50, T=101] (整数)
    ↓ post_quant_embed
       [B=50, T=101, D=256]
    ↓ decoder (4 layers)
       [B=50, T=101, D=256]
    ↓ head
输出:  [B=50, T=101, F=6]
```

**Predictor训练阶段**:
```
Token输入: s1_ids=[B=50, T=100], s2_ids=[B=50, T=100]
时间输入:  stamp=[B=50, T=100, F=5]
    ↓ HierarchicalEmbedding
       [B=50, T=100, D=512]
    ↓ + TemporalEmbedding
       [B=50, T=100, D=512]
    ↓ transformer (12 layers, causal)
       [B=50, T=100, D=512]
    ↓ norm
       [B=50, T=100, D=512]
    ↓ head (s1)
       s1_logits: [B=50, T=100, V1=1024]
    ↓ dependency_aware_layer + head (s2)
       s2_logits: [B=50, T=100, V2=1024]
```

**推理阶段 (自回归)**:
```
初始: x_seq = [B=1, T_hist=90, F=6]
      x_stamp = [B=1, T_hist=90, 5]
      y_stamp = [B=1, T_pred=10, 5]  # 未来时间戳

1. tokenizer.encode(x_seq) -> [s1_seq, s2_seq] each [B, 90]

for t in range(10):  # 预测10步
    # 当前序列长度: 90 + t
    if 90+t <= max_context(512):
        input_tokens = [s1_seq, s2_seq]
    else:
        input_tokens = [s1_seq[:,-512:], s2_seq[:,-512:]]  # 截断
    
    current_stamp = concat([x_stamp, y_stamp[:,:t+1,:]])
    
    # 预测下一个token
    s1_logits, ctx = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
    s1_new = sample_from_logits(s1_logits[:, -1, :])  # 采样最后一个位置
    
    s2_logits = model.decode_s2(ctx, s1_new)
    s2_new = sample_from_logits(s2_logits[:, -1, :])
    
    # 扩展序列
    s1_seq = concat([s1_seq, s1_new], dim=1)  # [B, 90+t+1]
    s2_seq = concat([s2_seq, s2_new], dim=1)

# 解码
preds = tokenizer.decode([s1_seq[:,-10:], s2_seq[:,-10:]])  # [B, 10, 6]
```

---

## 📊 损失函数设计

### 1. Tokenizer总损失

```python
total_loss = (recon_loss + bsq_loss) / 2

where:
    recon_loss = MSE(z_pre, x) + MSE(z, x)
    bsq_loss = commit_loss + zeta * entropy_penalty
    
    commit_loss = beta * MSE(z_quantized, z_before_quantize)
    entropy_penalty = gamma0 * H_per_sample - gamma * H_codebook
```

**各部分作用**:
- `MSE(z_pre, x)`: 确保s1_bits足够表达粗粒度信息
- `MSE(z, x)`: 确保完整codebook准确重建
- `commit_loss`: 使编码器输出接近量化后的值
- `H_per_sample`: 鼓励每个样本使用多样的码字（高熵）
- `H_codebook`: 鼓励整体码本均匀使用（高熵）

### 2. Predictor总损失

```python
total_loss = (CE_s1 + CE_s2) / 2

where:
    CE_s1 = CrossEntropy(s1_logits, s1_targets)
    CE_s2 = CrossEntropy(s2_logits, s2_targets)
```

**设计理由**:
- 分层预测：先预测粗粒度s1，再条件预测细粒度s2
- 平等权重：s1和s2同等重要
- 交叉熵：标准的分类损失，适合离散token预测

---

## 🚀 推理流程

### KronosPredictor 完整推理API

```python
class KronosPredictor:
    def __init__(self, model, tokenizer, device, max_context=512, clip=5):
        self.tokenizer = tokenizer.to(device)
        self.model = model.to(device)
        self.max_context = max_context
        self.clip = clip
    
    def predict(self, df, x_timestamp, y_timestamp, pred_len,
                T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        """
        Args:
            df: 历史数据 DataFrame [open, high, low, close, volume, amount]
            x_timestamp: 历史时间戳
            y_timestamp: 未来时间戳（预测目标）
            pred_len: 预测步数
            T: 采样温度（控制随机性）
            top_k: Top-K采样
            top_p: Nucleus采样
            sample_count: 多次采样求平均
        
        Returns:
            pred_df: 预测结果 DataFrame
        """
        # 1. 数据预处理
        x = df[price_cols + vol_cols].values  # [T, 6]
        x_stamp = calc_time_stamps(x_timestamp)  # [T, 5]
        y_stamp = calc_time_stamps(y_timestamp)  # [pred_len, 5]
        
        # 2. 归一化
        x_mean, x_std = mean(x), std(x)
        x = (x - x_mean) / (x_std + 1e-5)
        x = clip(x, -self.clip, self.clip)
        
        # 3. 自回归生成
        preds = auto_regressive_inference(
            self.tokenizer, self.model,
            x, x_stamp, y_stamp,
            self.max_context, pred_len,
            self.clip, T, top_k, top_p, sample_count, verbose
        )  # [1, pred_len, 6]
        
        # 4. 反归一化
        preds = preds * (x_std + 1e-5) + x_mean
        
        # 5. 构造DataFrame
        pred_df = DataFrame(preds[0], columns=cols, index=y_timestamp)
        return pred_df
```

### 自回归推理核心逻辑

```python
def auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp,
                               max_context, pred_len, clip, T, top_k, top_p, 
                               sample_count, verbose):
    batch_size, initial_seq_len, feat_dim = x.shape
    
    # 1. 多样本采样（用于uncertainty estimation）
    x = repeat(x, sample_count)  # [B*sample_count, T, 6]
    x_stamp = repeat(x_stamp, sample_count)
    y_stamp = repeat(y_stamp, sample_count)
    
    # 2. 编码初始序列
    x_token = tokenizer.encode(x, half=True)  # [s1_seq, s2_seq]
    
    # 3. 逐步生成
    for i in range(pred_len):
        current_seq_len = initial_seq_len + i
        
        # 3.1 处理上下文窗口
        if current_seq_len <= max_context:
            input_tokens = x_token
        else:
            input_tokens = [t[:, -max_context:] for t in x_token]  # 滑动窗口
        
        # 3.2 构造当前时间戳
        current_stamp = concat([x_stamp, y_stamp[:, :i+1, :]], dim=1)
        if current_stamp.shape[1] > max_context:
            current_stamp = current_stamp[:, -max_context:, :]
        
        # 3.3 预测s1
        s1_logits, context = model.decode_s1(
            input_tokens[0], input_tokens[1], current_stamp
        )
        s1_logits = s1_logits[:, -1, :]  # 取最后一个位置的logits
        
        # 3.4 采样s1
        sample_s1 = sample_from_logits(
            s1_logits, temperature=T, top_k=top_k, top_p=top_p
        )  # [B*sample_count, 1]
        
        # 3.5 预测s2（条件于s1）
        s2_logits = model.decode_s2(context, sample_s1)
        s2_logits = s2_logits[:, -1, :]
        
        # 3.6 采样s2
        sample_s2 = sample_from_logits(
            s2_logits, temperature=T, top_k=top_k, top_p=top_p
        )
        
        # 3.7 扩展token序列
        x_token[0] = concat([x_token[0], sample_s1], dim=1)
        x_token[1] = concat([x_token[1], sample_s2], dim=1)
    
    # 4. 解码最后pred_len步
    input_tokens = [t[:, -max_context:] for t in x_token]
    z = tokenizer.decode(input_tokens, half=True)  # [B*sample_count, T, 6]
    
    # 5. 重塑并平均多个样本
    z = z.reshape(batch_size, sample_count, -1, feat_dim)
    preds = mean(z, axis=1)  # [B, T, 6]
    
    return preds[:, -pred_len:, :]  # 只返回预测部分
```

### 采样策略

```python
def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None):
    # 1. 温度缩放
    logits = logits / temperature  # T↑→分布更平滑（多样性↑）
    
    # 2. Top-K过滤
    if top_k > 0:
        # 只保留概率最高的k个token
        threshold = torch.topk(logits, top_k)[0][..., -1, None]
        logits[logits < threshold] = -inf
    
    # 3. Nucleus (Top-P) 过滤
    if top_p < 1.0:
        # 保留累积概率达到p的最小token集合
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(softmax(sorted_logits), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        logits[sorted_indices_to_remove] = -inf
    
    # 4. 采样
    probs = softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    
    return sampled
```

---

## 🎨 架构设计亮点

### 1. 分层量化 (Hierarchical Quantization)

**动机**: 单一量化层难以同时捕捉粗粒度趋势和细粒度波动。

**实现**:
- `s1_bits` (10位): 捕捉主要趋势，词汇表大小 2^10=1024
- `s2_bits` (10位): 捕捉细节信息，词汇表大小 2^10=1024
- 总码本: 2^20 ≈ 100万个可能的组合

**优势**:
- 粗细结合：s1提供全局信息，s2补充局部细节
- 训练稳定：分层预测降低搜索空间复杂度
- 可解释性：s1可视化宏观趋势，s2显示微观调整

### 2. 条件依赖建模 (Dependency-Aware Layer)

**动机**: s2的预测应该依赖于s1的值（不独立）。

**实现**: 使用交叉注意力机制：
```python
x2 = DependencyAwareLayer(
    hidden_states=transformer_output,  # 上下文信息
    sibling_embed=s1_embedding         # s1的嵌入作为条件
)
s2_logits = head.cond_forward(x2)
```

**效果**: P(s2|s1, context) 而非 P(s2|context)

### 3. 旋转位置编码 (RoPE)

**优势对比传统位置编码**:
- **外推性**: 可以处理比训练时更长的序列
- **相对位置**: 自然编码token之间的相对距离
- **效率**: 不需要额外的参数或嵌入表

### 4. Pre-Norm Transformer

**对比Post-Norm**:
```
Post-Norm (原始Transformer):
x = x + Sublayer(x)
x = LayerNorm(x)

Pre-Norm (Kronos):
x = x + Sublayer(LayerNorm(x))
```

**优势**:
- 训练更稳定（梯度流更平滑）
- 可以训练更深的网络
- 收敛速度更快

### 5. SwiGLU激活函数

**对比ReLU/GELU**:
```python
ReLU:  f(x) = max(0, W1*x)
GELU:  f(x) = x * Φ(x)
SwiGLU: f(x) = (W1*x ⊙ σ(W2*x)) * W3
```

**优势**: 多项研究表明SwiGLU在Transformer中性能最佳

### 6. 实例级归一化

**对比全局归一化**:
```python
# 全局归一化（不适合金融数据）
x_normalized = (x - global_mean) / global_std

# 实例级归一化（Kronos采用）
x_normalized = (x - sample_mean) / sample_std
```

**原因**: 金融数据的尺度在不同时期差异巨大（如股价涨跌10倍），实例级归一化使模型关注**相对变化**而非绝对值。

---

## 📈 模型规模与参数量

### Tokenizer参数量估算

```python
# 配置: d_in=6, d_model=256, n_heads=8, ff_dim=1024, n_layers=4, s1+s2=20

embed:            6 * 256 = 1,536
encoder (4层):    4 * [
    attn: 256*256*4 (Q,K,V,O) = 262,144
    ffn: 256*1024*2 + 1024*256 = 786,432
] = 4,194,304

quant_embed:      256 * 20 = 5,120
BSQ: 0 (无可学习参数)

post_quant_embed_pre: 10 * 256 = 2,560
post_quant_embed:     20 * 256 = 5,120

decoder (4层):    4 * [同encoder] = 4,194,304

head:             256 * 6 = 1,536

总计: ≈ 8.4M 参数
```

### Predictor参数量估算

```python
# 配置: s1=10, s2=10, d_model=512, n_heads=8, ff_dim=2048, n_layers=12

HierarchicalEmbedding:
    emb_s1: 1024 * 512 = 524,288
    emb_s2: 1024 * 512 = 524,288
    fusion: 1024 * 512 = 524,288

TemporalEmbedding:
    minute/hour/weekday/day/month: (60+24+7+32+13) * 512 = 69,632

Transformer (12层): 12 * [
    attn: 512*512*4 = 1,048,576
    ffn: 512*2048*2 + 2048*512 = 3,145,728
] = 50,331,648

DependencyAwareLayer:
    cross_attn: 512*512*4 = 1,048,576

DualHead:
    proj_s1: 512 * 1024 = 524,288
    proj_s2: 512 * 1024 = 524,288

总计: ≈ 54M 参数
```

### 不同规模对比

| 模型 | d_model | n_layers | 参数量 | 训练时间(估计) | 推理速度 |
|------|---------|----------|--------|---------------|---------|
| Kronos-mini | 256 | 6 | ~20M | 快 | 很快 |
| Kronos-small | 512 | 12 | ~54M | 中 | 快 |
| Kronos-base | 768 | 16 | ~120M | 慢 | 中 |
| Kronos-large | 1024 | 24 | ~250M | 很慢 | 慢 |

---

## 🔍 关键超参数说明

### Tokenizer超参数

| 参数 | 典型值 | 作用 | 调优建议 |
|------|-------|------|---------|
| `s1_bits` | 10 | 粗粒度量化位数 | 增大→表达能力↑，计算量↑ |
| `s2_bits` | 10 | 细粒度量化位数 | 增大→细节捕捉↑，过拟合风险↑ |
| `beta` | 0.25 | 承诺损失权重 | 增大→量化更接近原始编码 |
| `gamma0` | 1.0 | 样本熵权重 | 增大→鼓励每个样本多样性 |
| `gamma` | 1.0 | 码本熵权重 | 增大→鼓励码本均匀使用 |
| `zeta` | 1.0 | 总熵正则权重 | 增大→熵正则化更强 |
| `group_size` | 9-10 | 熵计算分组大小 | 必须整除codebook_dim |

### Predictor超参数

| 参数 | 典型值 | 作用 | 调优建议 |
|------|-------|------|---------|
| `n_layers` | 12 | Transformer层数 | 增大→表达能力↑，训练难度↑ |
| `d_model` | 512 | 隐藏层维度 | 增大→参数量平方增长 |
| `n_heads` | 8 | 注意力头数 | 通常为d_model的约数 |
| `ff_dim` | 2048 | FFN维度 | 通常为d_model的4倍 |
| `token_dropout_p` | 0.1 | Token dropout概率 | 防止过拟合 |
| `max_context` | 512 | 最大上下文长度 | 受显存限制 |

### 训练超参数

| 参数 | Tokenizer | Predictor | 说明 |
|------|-----------|-----------|------|
| `learning_rate` | 2e-4 | 4e-5 | Predictor更小避免遗忘 |
| `batch_size` | 50 | 50 | 受显存限制 |
| `epochs` | 30 | 30 | 根据收敛情况调整 |
| `weight_decay` | 0.1 | 0.1 | L2正则化 |
| `grad_clip` | 2.0 | 3.0 | 梯度裁剪防止爆炸 |
| `accumulation_steps` | 1 | 1 | 模拟更大batch size |

### 推理超参数

| 参数 | 典型值 | 作用 | 效果 |
|------|-------|------|------|
| `temperature (T)` | 0.6-1.0 | 采样温度 | 小→确定性↑，大→多样性↑ |
| `top_k` | 0 | Top-K采样 | 0表示不使用 |
| `top_p` | 0.9 | Nucleus采样 | 截断长尾低概率token |
| `sample_count` | 5 | 多次采样平均 | 增大→预测更平滑，推理更慢 |

---

## 🎓 总结

### 核心创新点

1. **金融时序的离散化表示**: 借鉴NLP成功经验，将连续序列转为token
2. **分层量化策略**: s1捕捉趋势，s2捕捉细节
3. **条件依赖建模**: s2显式依赖s1
4. **二值球面量化**: 高效的离散化方案
5. **实例级归一化**: 适应金融数据的尺度变化

### 适用场景

✅ **适合**:
- 高频金融数据（分钟/小时级）
- 具有周期性的时间序列
- 需要多步预测
- 有充足训练数据

❌ **不适合**:
- 低频数据（月度/年度）
- 数据量极小（<1000样本）
- 纯随机游走序列
- 需要可解释性的场景（黑盒模型）

### 后续优化方向

1. **多模态融合**: 加入新闻、社交媒体等文本信息
2. **因果注意力改进**: 引入图结构建模股票关联
3. **元学习**: 快速适应新股票/新市场
4. **不确定性量化**: 提供预测置信区间
5. **在线学习**: 增量更新模型适应市场变化

---

**文档版本**: v1.0  
**生成时间**: 2025-10-15  
**作者**: AI Analysis Tool  
**项目**: Kronos Time Series Prediction Framework

