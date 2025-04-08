import os

# 设置缓存路径
os.environ["HF_HOME"] = "E:/my_huggingface_cache"

# tensor_parallel_baseline.py
# Python + PyTorch + Hugging Face + mpi4py 实现 Hugging Face 模型张量并行的 baseline

from transformers import AutoTokenizer
import torch
import torch.nn as nn
from mpi4py import MPI
import time
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置 tokenizer
model_id = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 手动构造一个 attention 模块（简化版本）
class ParallelSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 仅初始化自己负责的部分权重（模拟模型切分）
        self.q_proj = nn.Linear(embed_dim, embed_dim // world_size, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // world_size, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // world_size, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim // world_size, bias=False)

    def forward(self, x):
        # x: [B, T, C]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 简化 attention: QK^T / sqrt(d) @ V
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        out = self.out_proj(attn_output)  # 每个 rank 只输出一部分
        return out

# 初始化 attention 模块
embed_dim = 768
num_heads = 12
attention = ParallelSelfAttention(embed_dim, num_heads).to(device)
attention.eval()

# 准备输入
input_text = "The meaning of life is"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
input_embed = nn.Embedding(50257, embed_dim).to(device)(input_ids)

# 时间测量
start_time = time.time()
with torch.no_grad():
    local_out = attention(input_embed)
    local_out_np = local_out.cpu().numpy()

# Allgather
recv_shape = list(local_out_np.shape)
recv_shape[-1] *= world_size
recv_tensor = np.zeros(recv_shape, dtype=np.float32)

comm.Allgather([local_out_np, MPI.FLOAT], [recv_tensor, MPI.FLOAT])
total_time = time.time() - start_time

if rank == 0:
    print("==== Parallel Attention Allgather Result ====")
    print("Merged output shape:", recv_tensor.shape)
    print(f"Total time (compute + comm): {total_time:.4f}s")