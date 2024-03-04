import torch
import torch.distributed as dist 
import torch
import threading
from heterogeneous_utils import KVCacheModel,sample,norm_logits
import time
import numpy as np
import json
import zmq

context = zmq.Context()
tensor_socket = context.socket(zmq.REP)
tensor_socket.bind("tcp://*:1919")

tensor_lock = threading.Lock()
# shared_tensor = None

draft_tokens = None

from transformers import AutoTokenizer, AutoModelForCausalLM

target_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b",torch_dtype="auto", trust_remote_code=True)
target_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", trust_remote_code=True)

target_model.to('cuda:0')

@torch.no_grad()
def server_speculative_sampling_without_kvcache(draft_tokens : torch.Tensor, 
                        model : torch.nn.Module, 
                        temperature : float = 1, top_k : int = 0, 
                        top_p : float = 0, verbose : bool = False, 
                        random_seed : int = 1234) -> torch.Tensor:
    
    target_model_history = model(draft_tokens).logits
    for i in range(target_model_history.shape[-2]):
        target_model_history[:,i,:] = norm_logits(target_model_history[:,i,:],temperature,top_k,top_p)
    
    # shape_list = list(target_model_history.size())
    return target_model_history


def send_tensor_zeroMQ():
    """
    Received draft tokens via ZeroMQ and return an acknowledgment.
    """
    global draft_tokens
    message = tensor_socket.recv()  # Receive message
    data = json.loads(message.decode('utf-8'))  # Decode JSON data
    received_draft_tokens = torch.tensor(data['draft_tokens'])
    with tensor_lock:
        draft_tokens = received_draft_tokens
    tensor_socket.send_string('server received tokens')  # Send acknowledgment

def get_tensor_zeroMQ():
    """
    Send target model's logits via ZeroMQ.
    """
    global draft_tokens
    with tensor_lock:
        if draft_tokens is not None:
            list_to_tensor_time = time.time()
            draft_tokens_tensor = torch.tensor(draft_tokens).to("cuda:0")
            finish_list_to_tensor_time = time.time()

            target_forward_time = time.time()
            target_model_history_tensor = server_speculative_sampling_without_kvcache(
                draft_tokens=draft_tokens_tensor,
                model=target_model
            )
            finish_target_forward_time = time.time()

            tensor_to_list_time = time.time()
            target_model_history = target_model_history_tensor.to('cpu')
            model_history_list = target_model_history.tolist()
            finish_tensor_to_list_time = time.time()

            draft_tokens = None
            # Send the response as a JSON string
            response = json.dumps({
                'target_prob_hist': model_history_list,
                'tensor_to_list_time':finish_tensor_to_list_time - tensor_to_list_time,
                'target_model_generation_time':finish_target_forward_time-target_forward_time,
                'list_to_tensor_time': finish_list_to_tensor_time -list_to_tensor_time 
            })
            tensor_socket.send_string(response)
        else:
            tensor_socket.send_string(json.dumps({'target_prob_hist': None}))


if __name__ == "__main__":
    # Pre-run the target model
    input_ids = target_tokenizer.encode("Please write an introduction about UC Irvine: ", return_tensors='pt')
    input_ids = input_ids.to("cuda:0")
    _ = target_model(input_ids)
    # Enter a loop to continuously service the ZeroMQ REP socket
    while True:
        # Wait for next request from client
        request = tensor_socket.recv()
        if request.decode('utf-8') == 'send_tensor':
            send_tensor_zeroMQ()
        elif request.decode('utf-8') == 'get_tensor':
            get_tensor_zeroMQ()
        else:
            # Handle unexpected requests appropriately
            tensor_socket.send_string('unknown request')