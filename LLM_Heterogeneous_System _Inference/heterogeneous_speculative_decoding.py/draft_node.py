
# from torch.multiprocessing import Process
from  time import sleep
import datetime
import torch


import requests
import torch
import time
from heterogeneous_utils import KVCacheModel, sample, max_fn
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

    ## variables need send to server for kv_cache: 
    target_past_kv = None
    target_model_history = None 

    ## prefix = shape (1,prefix.len)
    while prefix.shape[1] < T: 
        # update prefix len, it will update during each iteration
        prefix_len = prefix.shape[1]
        draft_tokens = approx_model_cache.generate(prefix, gamma)
        # draft_tokens_to_server = draft_tokens.to('cpu')
        # draft_token_list = draft_tokens_to_server.tolist()
        
        #target_model_history and target_past_kv will be only None for the first time communicate to server
        send_tensor_to_server(SERVER_IP=SERVER_IP,
                              draft_tokens=draft_tokens,
                              update_prob=target_model_history,
                              past_kv=target_past_kv)
        
        #/get_tensor_from_server
        # get the 
            # 1. prob_history for comparison 
            # 2. past_kv for rollback in edge side
        received_dict = get_tensor(SERVER_IP=SERVER_IP)

        #############
        # how to wait until I get new tensor? 
        # I json error from get_tensor
        #############
        while received_dict is None: 
            print(f'wating................. nothing received from server')
            received_dict = get_tensor(SERVER_IP=SERVER_IP)
        
        target_model_history = received_dict['prob_history']
        target_past_kv = received_dict['past_kv']

        target_model_history = target_model_history.to('cuda:0')
        
        # don't need put the past kv on edge to device it is a list
        target_past_kv = target_past_kv

        # n for rollback endposition
        n = prefix_len + gamma - 1
        for i in range(gamma):
            # if random_seed:
            #     torch.manual_seed(random_seed)
            r = torch.rand(1, device = 'cuda:0')
            j = draft_tokens[:, prefix_len + i]
            
            if r > (target_model_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                # reject
                n = prefix_len + i - 1
                break  
            accepted_count += 1
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = draft_tokens[:, :n + 1]
        approx_model_cache.rollback(n+1)
        assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            """
            reject someone, sample from the pos n
            """
            t = sample(max_fn(target_model_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            resample_count += 1

            # update_target_cache(SERVER_IP,n+1)
            # doing the roll back on edge device
            rollback_dict = rollback(past_kv=target_past_kv,
                                     prob_hist=target_model_history,
                                     end_pos=n+1)
            # update kv and prob 
            target_past_kv = rollback_dict['past_kv']
            target_model_history = rollback_dict['prob_hist']

            # target_model_cache.rollback(n+1)
        else:
            """
            all approx model decoding accepted
            """
            assert n == target_model_history.shape[1] - 1
            t = sample(target_model_history[:, -1, :])
            target_sample_count += 1

            # update_target_cache(SERVER_IP,n+2)
            rollback_dict = rollback(past_kv=target_past_kv,
                                     prob_hist=target_model_history,
                                     end_pos=n+1)
            # update kv and prob 
            target_past_kv = rollback_dict['past_kv']
            target_model_history = rollback_dict['prob_hist']
        
        prefix = prefix.to("cuda:0")
        prefix = torch.cat((prefix, t), dim=1)

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
    end_time = time.time()
    print(f"Token Generation Speed (with speculative decoding): {max_len/(end_time-start_time)} tokens/s")
    print(f"Acceptance Rate: {accepted_count/max_len}")
    return prefix

import numpy as np
def rollback(past_kv,prob_hist,end_pos:int):
    
    # print(f'type of end_pos {type(end_pos)}, content of end_pos {end_pos}')
    past_key_values_trimmed = []
    for kv in past_kv:
        k, v = kv
        # NOTE() the indexing is specific for bloom. This won't work for other models
        # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
        
        # k, v (batch, head, seq, hidden_dim)
        # assert type(end_pos) ==int, 'end post must be int'
        
        k = k[:, :, :end_pos, :]
        v = v[:, :, :end_pos, :]
        kv_trimmed = (k, v)
        past_key_values_trimmed.append(kv_trimmed)
    past_kv = past_key_values_trimmed
    prob_hist = prob_hist[:, :end_pos, :]
    # self._prob_history = self._prob_history[:, :end_pos, :]
    return {'past_kv':past_kv,'prob_hist':prob_hist}

###############################################################################
# Replace with the actual IP address of the server
# def update_target_cache(SERVER_IP,port,index):
#     data = {'index': index}
#     response = requests.post(f'http://{SERVER_IP}:{port}/update_cache', json=data)
PORT = 5000
def send_tensor_to_server(SERVER_IP,
                          draft_tokens,
                          past_kv,
                          update_prob,
                          port = PORT):
    """
    correspond to send_tensor() function in server
    need make past_kv and update_prob to list
    """
    # handle kv
    if past_kv !=None: 
        past_key_values_list = []
        for kv in past_kv:
            k, v = kv
            k = k.tolist()
            v = v.tolist()
            past_key_values_list.append((k,v))
        past_kv = past_key_values_list
        update_prob = update_prob.to('cpu').tolist()

    data = {'draft_tokens': draft_tokens.to('cpu').tolist(),
            'past_kv':past_kv,
            'update_prob':update_prob}
    response = requests.post(f'http://{SERVER_IP}:{port}/send_tensor_to_server', json=data)
    print('send from edge to server',response.json())

def get_tensor(SERVER_IP,port=PORT):
    """
    correspond to get_tensor() function in server
    return 
    1. tensored prob history 
    2. tensored target_kv 
    """
    response = requests.get(f'http://{SERVER_IP}:{port}/get_tensor_from_server')
    target_prob_hist = response.json()['target_prob_hist']
    target_past_kv = response.json()['target_past_kv']
    if target_prob_hist is not None:
        target_model_cache_history = torch.tensor(target_prob_hist)
        print(f'cache_history from target is {target_model_cache_history.shape}')
        if target_past_kv is not None: 
            # it may not be necessary to make kv to tensor in edge device
            # target_past_kv_tensor = torch.tensor(target_past_kv)
            processed_kv_list = []
            # need extra carefull about the past_kv
            for kv in target_past_kv:
                k,v = kv
                k = torch.tensor(k)
                v = torch.tensor(v)
                processed_kv_list.append((k,v))
            target_past_kv_tensor = processed_kv_list
        return {'prob_history':target_model_cache_history,
                'past_kv':target_past_kv_tensor}#(target_model_cache_history,target_past_kv)
    else:
        return None

if __name__ == '__main__':
    # SERVER_IP = '192.168.0.239'
    SERVER_IP = '192.168.0.132'
    approx_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype="auto", trust_remote_code=True)
    approx_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", trust_remote_code=True)
    input_ids = approx_tokenizer.encode("Please write an introduction about UC Irvine: ", return_tensors='pt')
    # input_ids = input_ids.to('cuda:1')
    # inputs = target_tokenizer("Please write an introduction about UC Irvine: ", return_tensors="pt", return_attention_mask=False)
    top_k = 20
    top_p = 0.9
    output = edge_speculative_sampling(
        prefix=input_ids,
        approx_model=approx_model,
        SERVER_IP= SERVER_IP,
        max_len=10,
        gamma=4,
    )
    print(f'output is {approx_tokenizer.batch_decode(output)}')
  

