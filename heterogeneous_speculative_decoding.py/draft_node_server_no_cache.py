
# from torch.multiprocessing import Process
from  time import sleep
import datetime
import torch


import requests
import torch
import time
from heterogeneous_utils import KVCacheModel, sample, max_fn
from transformers import AutoTokenizer, AutoModelForCausalLM

class stats:
    def __init__(self):
        self.time_spend_sending_message = 0
        self.time_spend_tensor_to_list = 0
        self.time_spend_list_to_tensor = 0
        self.time_spend_on_draft_model_generation = 0
        self.time_spend_on_target_model_foward = 0 

heterogeneous_stats = stats()

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
        
        draft_generate_start_time = time.time()
        draft_tokens = approx_model_cache.generate(prefix, gamma)
        draft_generate_end_time = time.time()
        heterogeneous_stats.time_spend_on_draft_model_generation += draft_generate_end_time - draft_generate_start_time
        
        tensor_to_list_time = time.time()
        draft_tokens_list = draft_tokens.to('cpu').tolist()
        finish_tensor_to_list_time = time.time()
        heterogeneous_stats.time_spend_tensor_to_list += finish_tensor_to_list_time - tensor_to_list_time
        
        #target_model_history and target_past_kv will be only None for the first time communicate to server
        send_tensor_start_time = time.time()
        send_tensor_to_server(SERVER_IP=SERVER_IP,
                              draft_tokens_list=draft_tokens_list)
        
        #/get_tensor_from_server
        #get the logits
        target_model_mesg_dict = get_tensor(SERVER_IP=SERVER_IP)
        send_tensor_end_time = time.time()
        
        target_model_history = target_model_mesg_dict['target_prob_hist']
        tensor_to_list_time = target_model_mesg_dict['tensor_to_list_time']
        target_model_generation_time = target_model_mesg_dict['target_model_generation_time']
        list_to_tensor_time = target_model_mesg_dict['list_to_tensor_time']
        total_time_in_server = tensor_to_list_time+target_model_generation_time+list_to_tensor_time
        heterogeneous_stats.time_spend_sending_message += send_tensor_end_time - send_tensor_start_time -total_time_in_server
        heterogeneous_stats.time_spend_on_target_model_foward += target_model_generation_time
        heterogeneous_stats.time_spend_list_to_tensor += list_to_tensor_time
        heterogeneous_stats.time_spend_tensor_to_list +=tensor_to_list_time
        
        create_tensor_time = time.time()
        target_model_history = torch.tensor(target_model_history)
        target_model_history = target_model_history.to('cuda:0')
        finish_create_tensor_time = time.time()
        heterogeneous_stats.time_spend_list_to_tensor += finish_create_tensor_time - create_tensor_time
        #############
        # how to wait until I get new tensor? 
        # I json error from get_tensor
        #############
        
        
        
        # don't need put the past kv on edge to device it is a list

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
        else:
            """
            all approx model decoding accepted
            """
            assert n == target_model_history.shape[1] - 1
            t = sample(target_model_history[:, -1, :])
            target_sample_count += 1
        
        prefix = prefix.to("cuda:0")
        prefix = torch.cat((prefix, t), dim=1)

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
    end_time = time.time()
    print(f'total time spend on heterogeneous speculative decoding: {end_time - start_time}')
    print(f"Token Generation Speed (with speculative decoding): {max_len/(end_time-start_time)} tokens/s")
    print(f"Acceptance Rate: {accepted_count/max_len}")
    return prefix


###############################################################################
# Replace with the actual IP address of the server
# def update_target_cache(SERVER_IP,port,index):
#     data = {'index': index}
#     response = requests.post(f'http://{SERVER_IP}:{port}/update_cache', json=data)
PORT = 5000
def send_tensor_to_server(SERVER_IP,
                          draft_tokens_list,
                          port = PORT):
    """
    correspond to send_tensor() function in server
    need make past_kv and update_prob to list
    """
    
    data = {'draft_tokens': draft_tokens_list}
    response = requests.post(f'http://{SERVER_IP}:{port}/send_tensor_to_server', json=data)
    print('send from edge to server',response.json())

def get_tensor(SERVER_IP,port=PORT):
    """
    correspond to get_tensor() function in server
    """
    response = requests.get(f'http://{SERVER_IP}:{port}/get_tensor_from_server')
    target_prob_hist = response.json()['target_prob_hist']
    if target_prob_hist is not None:
        
        return response.json()
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
        max_len=100,
        gamma=4,
    )
    print(f'total time on communication: {heterogeneous_stats.time_spend_sending_message}')
    print(f'total time on list to tensor (tensor() + to(cuda:0)): {heterogeneous_stats.time_spend_list_to_tensor}')
    print(f'total time on tensor to list (to cpu() + tolist()): {heterogeneous_stats.time_spend_tensor_to_list}')
    print(f'total time on target model foward: {heterogeneous_stats.time_spend_on_target_model_foward}')
    print(f'total time on draft model generation: {heterogeneous_stats.time_spend_on_draft_model_generation}')
    print(f'output is {approx_tokenizer.batch_decode(output)}')
  

