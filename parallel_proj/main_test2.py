import os

# 设置缓存路径
os.environ["HF_HOME"] = "E:/my_huggingface_cache"

# gptneo_mpi_baseline.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
# 每个 rank 加载完整模型（数据并行）
model_id = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()

# 分配不同输入给不同 rank（模拟批处理）
all_inputs = [
    "The meaning of life is",
    "In a distant galaxy,",
    "The stock market crashed because",
    "Once upon a time in a forest,",
]
local_input = all_inputs[rank % len(all_inputs)]

# 编码输入
inputs = tokenizer(local_input, return_tensors="pt").to(device)

# 时间测量
start_time = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=30)
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
end_time = time.time()

# 收集所有 rank 的输出
all_texts = comm.gather(gen_text, root=0)
all_times = comm.gather(end_time - start_time, root=0)

if rank == 0:
    print("==== GPT-Neo Inference Results ====")
    for i, (text, t) in enumerate(zip(all_texts, all_times)):
        print(f"[Rank {i}] {text}  ({t:.4f}s)")
