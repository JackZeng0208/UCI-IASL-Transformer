import torch
import torch.distributed as dist 
from flask import Flask, request, jsonify
import torch
import threading
from utils import KVCacheModel

app = Flask(__name__)
tensor_lock = threading.Lock()
shared_tensor = None

from transformers import AutoTokenizer, AutoModelForCausalLM

target_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b",torch_dtype="auto", trust_remote_code=True)
target_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", trust_remote_code=True)
target_model_cache = KVCacheModel(target_model, 1, 0, 0).to('cuda:0')
    

@torch.no_grad()
def server_speculative_sampling(draft_tokens : torch.Tensor, 
                        target_model_cache : torch.nn.Module, 
                        temperature : float = 1, top_k : int = 0, 
                        top_p : float = 0, verbose : bool = False, 
                        random_seed : int = None) -> list  :
    
    
    draft_tokens = draft_tokens.to("cuda:0")
    _ = target_model_cache.generate(draft_tokens,1)
    target_model_history = target_model_cache._prob_history
    target_model_history = target_model_history.to('cpu')
    # shape_list = list(target_model_history.size())
    return target_model_history.tolist()

@app.route('/update_cache', methods=['POST'])
def update_cache():
    decision = None
    rollback_num = request.get_json()
    if rollback_num is not None:
        target_model_cache.rollback(int(rollback_num['index']))

    return jsonify({'message': f'target cache updated successfully'})

@app.route('/send_tensor_to_server', methods=['POST'])
def send_tensor():
    global shared_tensor
    data = request.get_json()
    received_tensor = torch.tensor(data['tensor_list'])
    
    with tensor_lock:
        shared_tensor = received_tensor
    
    return jsonify({'message': f'Tensor received by server successfully'})

@app.route('/get_tensor_from_server', methods=['GET'])
def get_tensor():
    global shared_tensor
    with tensor_lock:
        if shared_tensor is not None:
            # clone_tensor = shared_tensor.clone().detach() ?
            draft_tokens_tensor = torch.tensor(shared_tensor)
            history = server_speculative_sampling(
                draft_tokens=draft_tokens_tensor,
                target_model_cache= target_model_cache,
            )
            tensor_to_send = history
            shared_tensor = None
            return jsonify({'tensor_list': tensor_to_send})
        else:
            return jsonify({'tensor_list': None})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    # pre run the target model
    input_ids = target_tokenizer.encode("Please write an introduction about UC Irvine: ", return_tensors='pt')
    input_ids = input_ids.to("cuda:0")
    _  = target_model_cache.generate(input_ids,1)
    
    
    # ipp = '192.168.0.146'
    # '192.168.0.239'
    ips = "192.168.0.239"
    # init_processes(0,2,'0.0.0.0',"1234")
    # print("connected")
    app.run(host=ips,port='6100')