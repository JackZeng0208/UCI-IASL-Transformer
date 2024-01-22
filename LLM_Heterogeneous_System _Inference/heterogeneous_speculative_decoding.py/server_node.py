import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import socket
ip_address = socket.gethostbyname(socket.gethostname())
print(ip_address)

def init_process():
    os.environ['MASTER_ADDR'] = '66.42.104.193'  # IP of 4090
    os.environ['MASTER_PORT'] = '8233'        # A chosen port
    dist.init_process_group(backend='nccl', rank=0, world_size=2)

def run_target_model():
    init_process()
    

if __name__ == "__main__":
    main()