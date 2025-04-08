import os

# 设置缓存路径
os.environ["HF_HOME"] = "E:/my_huggingface_cache"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from mpi4py import MPI
import numpy as np
import time

# MPI 初始化
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Megatron-style 张量并行 Linear 层
class MegatronLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, world_size, rank):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        self.local_out_features = self.out_features // world_size
        start = rank * self.local_out_features
        end = (rank + 1) * self.local_out_features

        self.weight = nn.Parameter(original_linear.weight.data[start:end].clone())
        self.bias = nn.Parameter(original_linear.bias.data[start:end].clone()) if original_linear.bias is not None else None

    def forward(self, x):
        local_output = nn.functional.linear(x, self.weight, self.bias)
        return allreduce_tensor(local_output)

# AllReduce 聚合
def allreduce_tensor(local_tensor: torch.Tensor):
    np_tensor = local_tensor.detach().cpu().numpy()
    comm.Allreduce(MPI.IN_PLACE, np_tensor, op=MPI.SUM)
    return torch.tensor(np_tensor).to(local_tensor.device)

# 替换 GPT-Neo 中的 Linear 层
def apply_megatron_style_parallel(model, world_size, rank):
    for block in model.transformer.h:
        block.attn.attention.q_proj = MegatronLinear(block.attn.attention.q_proj, world_size, rank).to(device)
        block.attn.attention.k_proj = MegatronLinear(block.attn.attention.k_proj, world_size, rank).to(device)
        block.attn.attention.v_proj = MegatronLinear(block.attn.attention.v_proj, world_size, rank).to(device)
        block.attn.attention.out_proj = MegatronLinear(block.attn.attention.out_proj, world_size, rank).to(device)

        block.mlp.c_fc = MegatronLinear(block.mlp.c_fc, world_size, rank).to(device)
        block.mlp.c_proj = MegatronLinear(block.mlp.c_proj, world_size, rank).to(device)

# 加载模型和 tokenizer
model_id = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()

apply_megatron_style_parallel(model, world_size, rank)

# 每个进程一个输入
texts = [
    "The meaning of life is",
    "In a distant galaxy,",
    "The stock market crashed because",
    "Once upon a time in a forest,",
]
local_text = texts[rank % len(texts)]
inputs = tokenizer(local_text, return_tensors="pt").to(device)

# 单步推理（非 generate）
with torch.no_grad():
    input_ids = inputs["input_ids"]
    input_embed = model.transformer.wte(input_ids)
    hidden_states = input_embed

    start = time.time()
    for block in model.transformer.h:
        attn_out, _ = block.attn(hidden_states)  # 解包返回值
        hidden_states = hidden_states + attn_out
        hidden_states = block.ln_1(hidden_states)

        mlp_out = block.mlp(hidden_states)
        hidden_states = hidden_states + mlp_out
        hidden_states = block.ln_2(hidden_states)

    logits = model.lm_head(hidden_states)
    end = time.time()

    pred_token = torch.argmax(logits[0, -1])
    pred_word = tokenizer.decode(pred_token)

# 汇总输出
gathered_preds = comm.gather(pred_word, root=0)
gathered_times = comm.gather(end - start, root=0)

if rank == 0:
    print("\n==== Megatron-style Tensor Parallel GPT-Neo ====")
    for i in range(world_size):
        print(f"[Rank {i}] → '{texts[i]}' + '{gathered_preds[i]}'  ({gathered_times[i]:.4f}s)")
