import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import socket

"""
1. first test if double send and recv is working 
2. then test wither the while loop will also work
"""

print(torch.distributed.is_available())
print(torch.cuda.is_available())
print(torch.distributed.is_nccl_available())
print(socket.gethostname())

from transformers import AutoTokenizer, AutoModelForCausalLM

def get_distribution(logits, temperature):
    probs = torch.softmax(logits / (temperature + 1e-10), dim=-1)
#     print(f'probs return by get_distribution is {probs} shape is {probs.shape}')
    return probs

def sample(logits, temperature):
    probs = get_distribution(logits, temperature)
    return torch.multinomial(probs, num_samples=1)[0]

def sample_from_draft_model(
    model, 
    prefix, 
    gamma, 
    temperature=1.0
):
    # output_token = prefix.detach().clone()
    output_token = prefix
    out_logits = []
    
    for _ in range(gamma):
        sample_token_logits = model(output_token).logits[:, -1, :]
        sample_token = sample(sample_token_logits, temperature=temperature)
        # what is sample_token[None,...]??
        # print(f'what is sample_token[None,...]: {sample_token[None,...]}')
        # print(f'sample_token is {sample_token}')
        output_token = torch.concat([output_token, sample_token[None,...]], dim=-1)
        out_logits.append(sample_token_logits)

    out_logits = torch.stack(out_logits, dim=1)
#     print(f'output logit from draft m is {output_token} with shape of {output_token.shape}')
    return output_token, out_logits
            
            
    

def init_process():
    os.environ['MASTER_ADDR'] = '169.234.62.234'  # IP of 3060
    os.environ['MASTER_PORT'] = '8233'        # A chosen port
    dist.init_process_group(backend='nccl', rank=1, world_size=2)

def run_draft_model(
        draft_model,
        gamma,
        prefix,
        max_len =10,
):
    init_process()
    output_len = prefix.shape[-1]
    output = prefix # detech from device? 
    output = output.cuda()
    # while output_len < max_len:
        
    orig_output_len = output_len
    N = output.shape[-1]
#         print(f'check device draft model: {draft_model.device}, output: {output.device}')
    draft_outputs, draft_logits = sample_from_draft_model(draft_model,output,gamma)
    d1, d2 = draft_outputs.shape
    dist.send(tensor=torch.tensor((d1,d2)),dst=0)
    dist.send(tensor=draft_outputs,dst=0)

    # target logit shape is [1, gamma +1, world embedding dim]
    target_logit_shape = torch.empty((3),dtype=torch.long)
    dist.recv(tensor=target_logit_shape,src =0)
    print(f'darft side: what is the target logit shape {target_logit_shape}')
    d1, d2, d3 = target_logit_shape.shape
    target_logit = torch.empty((d1, d2, d3),dtype=torch.float32)
    dist.recv(tensor=target_logit,src=0)
    print(f'darft side: what is the target logit {target_logit}')




def heterogeneous_speculative_sampling(
    target_model,
    draft_model,
    prefix,
    max_len,
    tokenizer,
    gamma = 4,
    temperature = 1,
    debug = False
):
    output_len = prefix.shape[-1]
    output = prefix # detech from device? 
    output = output.to('cuda:1')
    while output_len < max_len:
        
        orig_output_len = output_len
        N = output.shape[-1]
#         print(f'check device draft model: {draft_model.device}, output: {output.device}')
        draft_outputs, draft_logits = sample_from_draft_model(draft_model,output,gamma)
        
        ##########################
        draft_outputs = draft_outputs.detach().to('cuda:0')
        # server cuda:0 only do one calculation and send the logit back to edge cuda:1
        target_logits = target_model(draft_outputs).logits[:,-gamma-1:,:]
        draft_outputs = draft_outputs.detach().to('cuda:1')
        target_logits = target_logits.detach().to('cuda:1')
        ###########################
        
        # result of the calculation should perform on cuda 1
        
        target_model_distribution = get_distribution(target_logits, temperature)
        draft_model_distribution = get_distribution(draft_logits, temperature)
#         print(f'check on cpu or gpu: target_model_distribution {target_model_distribution.device} , draft_model_distribution { draft_model_distribution.device}')
        if debug: 
            print(f"Possible continuations: {tokenizer.decode(draft_outputs[0,orig_output_len:], skip_special_tokens=True)}")
        accepted_flag = 1
        for i in range(gamma):
            numerator = target_model_distribution[:, i, draft_outputs[0, N+i]]
            denominator = draft_model_distribution[:, i, draft_outputs[0, N+i]]
            r = (numerator/denominator)
            uniform_distribution = torch.rand_like(numerator)
            ones_tensor = torch.ones_like(numerator)
            if (uniform_distribution < torch.min(ones_tensor,r)).any():
#                 print(f'check on cpu or gpu: draft_outputs {draft_outputs.device}, output { output.device}')
                output = torch.concat([output, draft_outputs[:, N+i].unsqueeze(dim=-1)], dim=-1)
                output_len += 1
            else:
                new_dist = (target_model_distribution[:, i, :] - draft_model_distribution[:, i, :])
                new_dist = torch.max(torch.zeros_like(new_dist), new_dist)
                new_dist = new_dist / new_dist.sum(dim=-1, keepdim=True)
                token_id = torch.multinomial(new_dist, num_samples=1)[0]
                output = torch.concat([output, token_id[None,...]], dim=-1)
                if debug:
                    print(f'correction is {tokenizer.decode(output[0,orig_output_len:], skip_special_tokens=True)}')
                accepted_flag = 0
                break
        if accepted_flag == 1:
            sample_token = sample(target_logits[:, -1, :], temperature=temperature)
            output = torch.concat([output, sample_token[None,...]], dim=-1)
        if debug:
            print(f"accepted continuations: {tokenizer.decode(output[0,orig_output_len:], skip_special_tokens=True)}")
        output_len += 1
    return output

if __name__ == "__main__":
    target_model = "facebook/opt-1.3b"
    draft_model = "facebook/opt-125m"


    d_model = AutoModelForCausalLM.from_pretrained(draft_model)
    d_tokenizer = AutoTokenizer.from_pretrained(draft_model)
    d_model.cuda()
    d_model.eval()
    prefix = d_tokenizer('once upon a time',return_tensors = 'pt').input_ids
    run_draft_model(d_model,4,prefix)