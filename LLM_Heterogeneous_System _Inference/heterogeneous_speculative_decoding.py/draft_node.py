
# from torch.multiprocessing import Process
from  time import sleep
import datetime
import torch


import requests
import torch
import time
from utilis import KVCacheModel, sample, max_fn
from transformers import AutoTokenizer, AutoModelForCausalLM

def edge_speculative_sampling(prefix : torch.Tensor, 
                         approx_model : torch.nn.Module, 
                         SERVER_IP: str,
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, 
                         verbose : bool = False, random_seed : int = None) -> torch.Tensor:
    """
    the edge speculative sample will handle
    1. place prefix to device
    2. place model to device
    """
    seq_len = prefix.shape[1]
    ## number of total token should generate
    T = seq_len+max_len
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p).to('cuda:0')

    #### stats collection ########
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    prefix = prefix.to('cuda:0')
    start_time = time.time()
    #### stats collection ########

    ## prefix = shape (1,prefix.len)
    while prefix.shape[1] < T: 
        # update prefix len, it will update during each iteration
        prefix_len = prefix.shape[1]
        draft_tokens = approx_model_cache.generate(prefix, gamma)
        draft_tokens_to_server = draft_tokens.to('cpu')
        # draft_token_shape = list(draft_tokens.size())
        draft_token_list = draft_tokens_to_server.tolist()
        send_tensor_to_server(SERVER_IP=SERVER_IP,tensor_list=draft_token_list)
        received_tensor = get_tensor(SERVER_IP=SERVER_IP)
        #############
        # how to wait until I get new tensor? 
        # I json error from get_tensor
        #############
        while received_tensor is None: 
            received_tensor = get_tensor(SERVER_IP=SERVER_IP)
        
        target_model_history = received_tensor
        target_model_history = target_model_history.to('cuda:0')
        n = prefix_len + gamma - 1
        

        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device = 'cuda:0')
            j = draft_tokens[:, prefix_len + i]
            # j = j.to('cpu')
            
            if r > (target_model_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                # reject
                n = prefix_len + i - 1
                break
            
            # if verbose:
            #     print(f"approx guess accepted {j[0]}: \033[31m{AutoTokenizer.decode(torch.tensor([j]))}\033[0m")

            accepted_count += 1
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = draft_tokens[:, :n + 1]
        
        approx_model_cache.rollback(n+1)
        
        assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            # reject someone, sample from the pos n
            t = sample(max_fn(target_model_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            # if verbose:
            #     print(f"target resamples at position {n}: \033[34m{tokenizer.decode(t)}\033[0m")
            resample_count += 1
            update_target_cache(SERVER_IP,n+1)
            # target_model_cache.rollback(n+1)
        else:
            # all approx model decoding accepted
            assert n == target_model_history.shape[1] - 1
            t = sample(target_model_history[:, -1, :])
            # if verbose:
            #     print(f"target samples {n}: \033[35m{tokenizer.decode(t)}\033[0m")
            target_sample_count += 1
            update_target_cache(SERVER_IP,n+2)
            # target_model_cache.rollback(n+2)
        prefix = prefix.to("cuda:0")
#         print(f'prefix device is {prefix.device}, t device is {t.device}')
        prefix = torch.cat((prefix, t), dim=1)

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
    end_time = time.time()
    print(f"Token Generation Speed (with speculative decoding): {max_len/(end_time-start_time)} tokens/s")
    print(f"Acceptance Rate: {accepted_count/max_len}")
    return prefix



###############################################################################
# Replace with the actual IP address of the server
def update_target_cache(SERVER_IP,index):
    data = {'index': index}
    response = requests.post(f'http://{SERVER_IP}:6100/update_cache', json=data)

def send_tensor_to_server(SERVER_IP, tensor_list):
    data = {'tensor_list': tensor_list}
    response = requests.post(f'http://{SERVER_IP}:6100/send_tensor_to_server', json=data)
    print('send from edge to server',response.json())

def get_tensor(SERVER_IP):
    response = requests.get(f'http://{SERVER_IP}:6100/get_tensor_from_server')
    tensor_data = response.json()['tensor_list']
    if tensor_data is not None:
        target_model_cache_history = torch.tensor(tensor_data)
        print(f'cache_history from target is {target_model_cache_history.shape}')
        return target_model_cache_history
    else:
        return None

if __name__ == '__main__':
    SERVER_IP = '192.168.0.239'
    approx_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype="auto", trust_remote_code=True)
    approx_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", trust_remote_code=True)
    input_ids = approx_tokenizer.encode("Please write an introduction about UC Irvine: ", return_tensors='pt')
    # input_ids = input_ids.to('cuda:1')
    # inputs = target_tokenizer("Please write an introduction about UC Irvine: ", return_tensors="pt", return_attention_mask=False)
    top_k = 20
    top_p = 0.9
    edge_speculative_sampling(
        prefix=input_ids,
        approx_model=approx_model,
        SERVER_IP= SERVER_IP,
        max_len=10,
        gamma=4,
    )
  

