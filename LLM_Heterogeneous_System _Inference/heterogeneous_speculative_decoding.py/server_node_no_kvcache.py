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

from transformers import AutoTokenizer, AutoModelForCausalLM

target_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b",torch_dtype="auto", trust_remote_code=True)
target_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", trust_remote_code=True)
# target_model_cache = KVCacheModel(target_model, 1, 0, 0).to('cuda:0')
target_model.to('cuda:0')
@torch.no_grad()
########################### new server_speculatie sampling 

###########################
@torch.no_grad()
def server_speculative_sampling_without_kvcache(draft_tokens : torch.Tensor, 
                        model : torch.nn.Module, 
                        temperature : float = 1, top_k : int = 0, 
                        top_p : float = 0, verbose : bool = False, 
                        random_seed : int = None) -> list  :
    
    
    draft_tokens = draft_tokens.to("cuda:0")
    target_model_history = model(draft_tokens).logits
    for i in range(target_model_history.shape[-2]):
        target_model_history[:,i,:] = norm_logits(target_model_history[:,i,:],temperature,top_k,top_p)
    target_model_history = target_model_history.to('cpu')
    # shape_list = list(target_model_history.size())
    return target_model_history.tolist()


@app.route('/send_tensor_to_server', methods=['POST'])
def send_tensor():
    """
    received draft tokens  
    return 
    1. tensored draft tokens
    """
    global draft_tokens
    data = request.get_json()
    # received_draft_tokens = torch.tensor(data['tensor_list'])
    received_draft_tokens = torch.tensor(data['draft_tokens'])
    
    with tensor_lock:
        draft_tokens = received_draft_tokens
    return jsonify({'message': f'server received tokens'})

@app.route('/get_tensor_from_server', methods=['GET'])
def get_tensor():
    '''
    send 
    1. target model's logits
    '''
    # global shared_tensor
    global draft_tokens

    with tensor_lock:
        if draft_tokens is not None:
            # clone_tensor = shared_tensor.clone().detach() ?
            draft_tokens_tensor = torch.tensor(draft_tokens)
            target_model_history_list = server_speculative_sampling_without_kvcache(
                draft_tokens=draft_tokens_tensor,
                model= target_model
            )
            
            draft_tokens = None
            # return jsonify({'tensor_list': tensor_to_send})
            return jsonify({'target_prob_hist': target_model_history_list})
        else:
            return jsonify({'target_prob_hist': None})

if __name__ == "__main__":
    # pre run the target model
    input_ids = target_tokenizer.encode("Please write an introduction about UC Irvine: ", return_tensors='pt')
    input_ids = input_ids.to("cuda:0")
    _ =target_model(input_ids)
    # ips = "192.168.0.239"
    IP = '0.0.0.0'
    # init_processes(0,2,'0.0.0.0',"1234")

    app.run(host=IP,port=5000)