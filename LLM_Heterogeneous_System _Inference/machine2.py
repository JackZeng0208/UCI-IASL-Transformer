# Machine 2: Responsible for receiving data from Machine 1, processing it, and optionally sending it back

import torch
import torch.distributed as dist
import os
from transformers import AutoModelForCausalLM

def init_process():
    os.environ['MASTER_ADDR'] = '128.195.55.253'  # Replace with the actual IP of Machine 1
    os.environ['MASTER_PORT'] = '8233'        # The same port as used on Machine 1
    dist.init_process_group(backend='nccl', rank=1, world_size=2)

def load_and_partition_model():
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", device_map="cuda", trust_remote_code=True)
    num_layers = len(model.transformer.h)
    model.transformer.h = model.transformer.h[num_layers // 2:]  # Keep the second half of the layers
    return model.to('cuda')

def main():
    init_process()
    model = load_and_partition_model()

    # Receive input from Machine 1
    received_tensor_size = (1, 512, 768)
    received_tensor = torch.empty(received_tensor_size, dtype=torch.long, device='cuda')
    dist.recv(tensor=received_tensor, src=0)

    with torch.no_grad():
        outputs = model(received_tensor)

    # Further processing of outputs can be done here

if __name__ == "__main__":
    main()
