import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

"""structure
'model'
1. single head attention
2. multi head attention
3. feed forward(mlp)
4. block
5. GPT
6. embedding, positional, norm, mlp, block
'data'
构建Dataset
"""

"""定义基本参数"""

@dataclass
class GPTConfig:
    block_size: int = 512  # 文本的最大长度，max_seq_len
    batch_size: int = 12
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768  # hidden_dim, hidden_size,为了可以tie_embedding_weight
    hidden_dim: n_embd
    dropout: float = 0.1
    head_size: int = n_embd // n_heads
    vocab_size: int = 50257  # GPT2官方的tokenizer的vocab_size


"""single head attention"""
class SingleHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig)
        super().__init__()
        self.key = nn.Linear(config.hidden_dim, config.head_size) 
        self.query = nn.Linear(config.hidden_dim, config.head_size)
        self.value = nn.Linear(config.hidden_dim, config.head_size)

        # attention_mask使用新写法，通过register_buffer注册，不用计算梯度，节约内存和显存，速度更快
        self.register_buffer(
            "attention_mask",
            torch.tril(
                torch.ones(config.block_size, config.bliock_size)
            )
        )

        # dropout提升泛化性，加速训练
        self.dropout = nn.dropout(config.dropout)
    
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        q, k, v = self.query(x), self.key(x), self.value(x)
        weight = q @ k.transpose(-2, -1) * (1/ math.sqrt(k.size(-1)))
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0,
            float("-inf")
        )
        weight = F.softmax(weight, dim = -1)
        weight = self.dropout(weight)
        output = weight @ v
        return output

"""multi head attention"""
class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.heads = nn.ModuleList(
            [SingleHeadAttention(config) for _ in config.n_heads]
        )

        # concact层
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.dropout(config.dropout)

    def forward(self, x):
        output = torch.cat(
            [h(x) for h in self.heads],
            dim = -1
        )
        output = self.proj(output)
        output = self.dropout(output)
        return output

"""feed forward"""
