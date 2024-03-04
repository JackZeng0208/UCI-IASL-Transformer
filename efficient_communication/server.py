import zmq
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


def send_tensor(tensor, port=1919):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)  # PUSH socket
    socket.bind(f"tcp://*:{port}")  # Bind to the given port

    payload = tensor.numpy().tobytes()

    # Send tensor shape and dtype as well (needed for reconstruction)
    metadata = {
        'dtype': str(tensor.dtype),
        'shape': tensor.size()
    }

    # Send metadata and payload
    socket.send_pyobj(metadata)
    socket.send(payload)

    socket.close()
    context.term()

# Example tensor to send
tensor = torch.randn(2, 2)  # Example tensor
print(tensor)
send_tensor(tensor)