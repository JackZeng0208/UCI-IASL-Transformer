import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed
import time
from tqdm import tqdm
torch.set_default_device(device='cuda')
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to('cuda:0')
model.eval()
prompt = ["Write an detailed introduction about UCI: "]

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
token_generation_speed = []
for _ in tqdm(range(100)):
    with torch.no_grad():
        start_time = time.time()
        generated_ids = model.generate(input_ids, do_sample = True, max_length = 500)
        end_time = time.time()
        token_generation_speed.append(len(generated_ids[0]) / (end_time - start_time))
print(f"Average token generation speed: {np.average(token_generation_speed)} tokens/s (Without DeepSpeed)")

# model = deepspeed.init_inference(model, dtype=torch.float16, replace_with_kernel_inject=True).to('cuda:0')
# model.eval()

# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
# token_generation_speed = []
# for _ in tqdm(range(100)):
#     with torch.no_grad():
#         start_time = time.time()
#         generated_ids = model.generate(input_ids, do_sample = True, max_length = 500)
#         end_time = time.time()
#         token_generation_speed.append(len(generated_ids[0]) / (end_time - start_time))
# print(f"Average token generation speed: {np.average(token_generation_speed)} tokens/s (With DeepSpeed)")