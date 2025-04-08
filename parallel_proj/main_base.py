import os

# 设置缓存路径
os.environ["HF_HOME"] = "E:/my_huggingface_cache"

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
print(model)
print(model.config.num_heads)