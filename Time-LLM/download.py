from transformers import GPT2Model, GPT2Tokenizer

model_name = "gpt2"  # 또는 "gpt2-medium", "gpt2-large", "gpt2-xl" 중 하나
output_dir = "/mnt/storage/personal/eungyeop/HERO/LLM/gpt2_model"  # 모델을 저장할 디렉토리

# 모델과 토크나이저 다운로드
model = GPT2Model.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 모델과 토크나이저 저장
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")


import os
import json
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig

# 원본 LLAMA 모델 경로
src_path = "/path/to/original/llama"
# 변환된 모델을 저장할 경로
dst_path = "/mnt/storage/personal/eungyeop/HERO/LLM/llama"

# 모델 크기 (예: 7B, 13B, ...)
model_size = "7B"

# 설정 파일 로드
with open(os.path.join(src_path, model_size, "params.json"), "r") as f:
    params = json.load(f)

# Hugging Face 설정 생성
config = LlamaConfig(
    hidden_size=params["dim"],
    intermediate_size=params["hidden_dim"],
    num_attention_heads=params["n_heads"],
    num_hidden_layers=params["n_layers"],
    rms_norm_eps=params["norm_eps"],
)

# 토크나이저 변환 및 저장
tokenizer = LlamaTokenizer.from_pretrained(src_path)
tokenizer.save_pretrained(dst_path)
print(f"Tokenizer saved to {dst_path}")

# 모델 로드 및 변환
model = LlamaForCausalLM(config)
state_dict = torch.load(os.path.join(src_path, model_size, "consolidated.00.pth"), map_location="cpu")
model.load_state_dict(state_dict, strict=False)

# 변환된 모델 저장
model.save_pretrained(dst_path)
print(f"Model saved to {dst_path}")

print("Conversion completed.")