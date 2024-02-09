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

def run_target_model(target_model,
                     gamma):
    init_process()
    # draft output: [1, prefix.len + gamma]
    draft_output_shape = torch.empty((2),dtype=torch.long)

    
    
    dist.recv(tensor=draft_output_shape, src=1)
    print(f'server side: what is the draft_output_shape {draft_output_shape}')
    draft_output = torch.empty((draft_output_shape[0],draft_output_shape[1]),dtype=torch.long)

    dist.recv(tensor=draft_output,src =1)
    print(f'server side: what is the draft_output {draft_output}')

    target_logits = target_model(draft_output).logits[:,-gamma-1:,:]
    d1, d2, d3 = target_logits.shape
    dist.send(tensor=torch.tensor((d1,d2,d3)),dst=1)
    dist.send(tensor=target_logits,dst=1)

    # first receive the shape from weak machine 
    # then assign the empty tensor with the shape
    # second receive the draft_output from weak machine
# run_target_model()
if __name__ == "__main__":
    target_model = "facebook/opt-1.3b"
    draft_model = "facebook/opt-125m"

    t_model = AutoModelForCausalLM.from_pretrained(target_model)
    t_tokenizer = AutoTokenizer.from_pretrained(target_model)
    t_model.cuda()
    t_model.eval()
    run_target_model(t_model,gamma=4)