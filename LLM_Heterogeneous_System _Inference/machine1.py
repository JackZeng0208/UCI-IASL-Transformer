# Imports
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Initialize the distributed environment
# def init_process():
#     os.environ['MASTER_ADDR'] = '128.195.55.253'  # IP of Machine 1
#     os.environ['MASTER_PORT'] = '8233'        # A chosen port
#     dist.init_process_group(backend='nccl', rank=0, world_size=2)

# init_process()
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
first_half_model = torch.nn.Sequential(*list(model.children())[:len(list(model.children())) // 2])

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

print(model.transformer)
# Send output to Machine 2
# dist.send(tensor=outputs, dst=1)
