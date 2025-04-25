import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import tiktoken
import json

"""structure
'model'
1. single head attention
2. multi head attention
3. feed forward(mlp)
4. block
5. GPT : embedding, position, block, norm, mlp, softmax
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
class FeedForwaed(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4)  # swiglu升到8/3
            nn.GELU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.dropout(config.dropout)
        )
    
    def forward(self, x):
        output = self.net(x)
        return output

"""block"""
class Block(nn.Module):
    def __init__(slef, config: GPTConfig):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ffn = FeedForwaed(config)
        self.ln1 = nn.Layernorm(config.hidden_dim)
        self.ln2 = nn.Layernorm(config.hidden_dim)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

"""GPT : embedding, position, block, norm, mlp, softmax"""
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        '''
        后续其他模型相对改动:
        position_embedding: 0,1,...,n -> repo
        norm: layer_norm -> rms_norm
        mlp -> swiglu
        mha -> gqa
        '''
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layers)]
        )
        self.ln_final = nn.Layernorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        '''注意: SLM模型使用tie_weight减少参数'''
        self.token_embedding_table.weight = self.lm_head.weight
    def _inint_weight(self, module):
        if isinstance(module, nn.Linear):
            # 初始化为正态分布
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zero_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets = None):
        """
        idx:输入的token ids
        targets:输出的token ids
        两者shape要一致
        """
        batch, seq_len = idx.size()
        token_emb = self.token_embedding_table(idx)
        position_emd = self.position_embedding_table(
            torch.arrange(seq_len, device=idx.device)
        )

        x = token_emb + position_emb  # [batch, seq_len, hidden_dim]
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        if targets is None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    # 生成当前词后，会将当前词加入输入序列作为新的token预测下一个词
    def generate(self, idx, max_new_tokens):
        # idx shape: (batch, seq_len)
        for _ in range(max_new_tokens):
            # 加入上一轮生成的新token后，若输入序列长度小于block_size，则继续使用idx作为输入，超出则截取idx[:, -block_size:]
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            # logits shape: (batch, seq_len, vocab_size)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # 随机采样
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) # (batch, seq_len+1)
        return idx

""" Dataset """

class MyDataset(Dataset):
    def __init__(self, path, blck_size = 512):
        # tiktoken 是GPT官方提供的tokenizer，使用GPU加速
        self.enc = titoken.get_encoding("gpt2")
        self.block_size = blck_size  # pos 最大长度

        self.encoded_data = []
        # 特殊符号<|endoftext|>分割不同的训练文本
        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allow_special = {"<|endoftext|>"}
        )

        # 读取数据，序列化
        self.max_lines = 1000
        raw_data = []  # 控制读取量
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    text = json.loads(lines.strip())["text"]
                    raw_data.append(text)
                except Exception as e:
                    print(e)
                    continue
        
        # encode, 将文本拼接
        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)  # list
            full_encoded.extend(encoded_text + [self.eos_token])
        
        # 拼接长度不能超过block_size,需进行分割
        for i in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[i : i + self.block_size + 1]  # 实际长度为513，因为计算loss时需要预测下一个token，将输入往右移一位
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)
        
    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype = torch.long)  # 前512作为输入
        y = torch.tensor(chunk[1:], dtype = torch.long)  # 后512作为输出，预测下一个词
        return x, y
        
    def encode(self, text):
        """将文本编码成token IDs"""
        return self.enc.encode(text)
        
    def decode(self, tokens):
        """将token IDs解码成文本"""
        return self.enc.decode(tokens)


"""Run"""

model = GPT(GPTConfig())
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 打印模型参数
total_params = sum(p.numel() for p in moedel.parameters())
print(f"Total parameters: {total_params / 1e6} M")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# 设置cosine学习率衰减
scheduler = torch.optim.lr_scheduler.CosineANNealingLR(optimizer, T_max=100)

"""Train"""
def train(model, optimizer, scheduler, train_loader, val_loader, device):
    model.train()
    for batch_idx, (x, y) in emunerate(tarin_loader):
        x, y = x.to(device), y.to(device)
        # 前向传播
        logits, loss = model(x, tagets=y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 学习率调整
        scheduler.step()

        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        return total_loss

def eval(model, val_loader, device):
    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            val_loss += loss.item()
    return val_loss

# train数据集
tarin_dataset = MyDataset('')
# 划分训练集和测试集
train_dataset, val_dataset = torch.utils.data.random_split(tarin_dataset, [0.8, 0.2])
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)


for epoch in range(2):
    train_loss = train(model, optimizer, scheduler, tarin_loader, device)
    val_loss = eval(model, val_loader, device)
    print('epoch: {}, train loss: {}, val loss: {}'.format(epoch, train_loss, val_loss))

    # 保存模型
    avg_val_loss = val_loss / len(val_loader)
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'avg_val_loss': avg_val_loss

    }
    torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')


"""推理"""
# 加载模型  
checkpoint = torch.load('checkpoints/model_epoch_4.pt')
model = GPT(GPTConfig())
model.load_state_dict(checkpoint['model'])
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# 准备初始输入
dataset = MyDataset('')
start_text = '我'
enc_token = dataset.encode(start_text)
input_ids = torch.tensor([enc_token], dtype=torch.long).to(device)
# 生成文本
generate_tokens = model.generate(input_idxs, max_length=100)
# 解码
generated_text = dataset.decode(generate_tokens[0].tolist())
print(generated_text)







