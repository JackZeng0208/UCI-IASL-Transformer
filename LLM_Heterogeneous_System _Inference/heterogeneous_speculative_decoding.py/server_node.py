import torch
import torch.distributed as dist 
from flask import Flask, request, jsonify
import torch
import threading
from heterogeneous_utils import KVCacheModel,sample,norm_logits

app = Flask(__name__)
tensor_lock = threading.Lock()
# shared_tensor = None

draft_tokens = None
past_kv = None
given_prob = None

from transformers import AutoTokenizer, AutoModelForCausalLM

target_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b",torch_dtype="auto", trust_remote_code=True)
target_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", trust_remote_code=True)
# target_model_cache = KVCacheModel(target_model, 1, 0, 0).to('cuda:0')
target_model.to('cuda:0')
@torch.no_grad()
########################### new server_speculatie sampling 
def new_server_speculative_sampling_with_kvCache(draft_tokens : torch.Tensor, 
                                    past_kv: torch.Tensor,
                                    given_prob_history,
                        model : torch.nn.Module,temperature : float = 1, top_k : int = 0, 
                        top_p : float = 0, verbose : bool = False, 
                        random_seed : int = None):
    """
    need return 
    1. probability history,
    2. past key values 
    """
    draft_tokens = draft_tokens.to("cuda:0")
    
    if past_kv is None: 

        outputs = model(draft_tokens)
        _prob_history = outputs.logits
        for i in range(_prob_history.shape[-2]):
            _prob_history[:,i,:] = norm_logits(_prob_history[:, i, :], temperature, top_k, top_p)
        _past_key_values = outputs.past_key_values

    elif past_kv: # already have previous kv 
        """
        need given probability hist
        """
        # past_kv and given_prob_history should both be none or not none
        _prob_history = given_prob_history.to('cuda:0')

        # past_kv need to device when received it 
        past_kv = past_kv
        assert _prob_history != None, "_prob_history shouldn't be None"
        cached_len = 0
        for kv in past_kv:
            k, v = kv
            cached_len = k.shape[2]
            
        last_input_id = draft_tokens[:, cached_len:]
        if last_input_id.dim() == 1:
            last_input_id = torch.unsqueeze(last_input_id, 0)
        outputs = model(last_input_id, past_key_values=past_kv, use_cache=True)
        not_cached_q = outputs.logits
        if not_cached_q.dim() == 2:
            not_cached_q = torch.unsqueeze(not_cached_q, 0)
            
        for i in range(not_cached_q.shape[-2]):   
            not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], temperature, top_k,top_p)  

        #  _prob_history should be given 
        _prob_history = torch.cat([_prob_history, not_cached_q], dim=1)
        
        _past_key_values = outputs.past_key_values
    _prob_history = _prob_history.to('cpu')
    ## _past_key_values is a tuple with 12 tensor to sure the keys values, each tensor store key and value 
    ## each k or v is size torch.Size([1, 12, 11, 64])
    past_key_values_list = []
    for kv in _past_key_values:
        k, v = kv
        k = k.to('cpu').tolist()
        v = v.to('cpu').tolist()
        past_key_values_list.append((k,v))
    _prob_history = _prob_history.to('cpu')
    # _past_key_values = _past_key_values.to('cpu')

    return {'prob_history':_prob_history.tolist(),
            'past_key_values':past_key_values_list}
###########################
# @torch.no_grad()
# def server_speculative_sampling(draft_tokens : torch.Tensor, 
#                         target_model_cache : torch.nn.Module, 
#                         temperature : float = 1, top_k : int = 0, 
#                         top_p : float = 0, verbose : bool = False, 
#                         random_seed : int = None) -> list  :
    
    
#     draft_tokens = draft_tokens.to("cuda:0")
#     _ = target_model_cache.generate(draft_tokens,1)
#     target_model_history = target_model_cache._prob_history
#     target_model_history = target_model_history.to('cpu')
#     # shape_list = list(target_model_history.size())
#     return target_model_history.tolist()

# @app.route('/update_cache', methods=['POST'])
# def update_cache():
#     decision = None
#     rollback_num = request.get_json()
#     if rollback_num is not None:
#         target_model_cache.rollback(int(rollback_num['index']))

#     return jsonify({'message': f'target cache updated successfully'})

@app.route('/send_tensor_to_server', methods=['POST'])
def send_tensor():
    """
    received draft tokens and past_kv from draft edge 
    return 
    1. tensored draft tokens
    2. tensored past_kv from draft edge 
    """
    global draft_tokens
    global past_kv 
    global given_prob
    data = request.get_json()
    # received_draft_tokens = torch.tensor(data['tensor_list'])
    received_draft_tokens = torch.tensor(data['draft_tokens'])
    if data['past_kv'] and data['update_prob']:
        processed_kv_list = []
        # need extra carefull about the past_kv
        for kv in data['past_kv']:
            k,v = kv
            k = torch.tensor(k,dtype=torch.float16)
            k = k.to('cuda:0')
            v = torch.tensor(v,dtype=torch.float16)
            v = v.to('cuda:0')
            processed_kv_list.append((k,v))
        received_past_kv = processed_kv_list
        received_given_prob = torch.tensor(data['update_prob'])
    else:
        received_past_kv = None
        received_given_prob = None
    with tensor_lock:
        draft_tokens = received_draft_tokens
        past_kv = received_past_kv
        given_prob = received_given_prob
    return jsonify({'message': f'server received draft_tokens, past_kv, update_prob'})

@app.route('/get_tensor_from_server', methods=['GET'])
def get_tensor():
    '''
    send 
    1. tokens and 
    2. kv 
    to edge device
    '''
    # global shared_tensor
    global draft_tokens
    global past_kv 
    global given_prob
    with tensor_lock:
        if draft_tokens is not None:
            # clone_tensor = shared_tensor.clone().detach() ?
            draft_tokens_tensor = torch.tensor(draft_tokens)

            # def new_server_speculative_sampling_with_kvCache(draft_tokens : torch.Tensor, 
            #                         past_kv: torch.Tensor,
            #                         given_prob_history,
            #             model : torch.nn.Module,temperature : float = 1, top_k : int = 0, 
            #             top_p : float = 0, verbose : bool = False, 
            #             random_seed : int = None):
            server_sampling_dict = new_server_speculative_sampling_with_kvCache(
                draft_tokens=draft_tokens_tensor,
                past_kv= past_kv,
                given_prob_history=given_prob,
                model = target_model
            )
            draft_tokens = None
            past_kv = None
            # return jsonify({'tensor_list': tensor_to_send})
            return jsonify({'target_prob_hist': server_sampling_dict['prob_history'],
                            'target_past_kv':server_sampling_dict['past_key_values']})
        else:
            return jsonify({'target_prob_hist': None,'target_past_kv':None})

if __name__ == "__main__":
    # pre run the target model
    input_ids = target_tokenizer.encode("Please write an introduction about UC Irvine: ", return_tensors='pt')
    input_ids = input_ids.to("cuda:0")
    _ =target_model(input_ids)
    # ips = "192.168.0.239"
    IP = '0.0.0.0'
    # init_processes(0,2,'0.0.0.0',"1234")

    app.run(host=IP,port=5000)