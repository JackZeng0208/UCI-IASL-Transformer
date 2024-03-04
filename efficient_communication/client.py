import zmq
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
def dtype_mapping(torch_dtype):
    """Map a PyTorch dtype to the equivalent NumPy dtype."""
    mapping = {
        'torch.float32': np.float32,
        'torch.float64': np.float64,
        'torch.float16': np.float16,
        'torch.uint8': np.uint8,
        'torch.int8': np.int8,
        'torch.int16': np.int16,
        'torch.int32': np.int32,
        'torch.int64': np.int64,
    }
    return mapping.get(str(torch_dtype), np.float32) 

def receive_tensor(port=1919):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)  # PULL socket
    socket.connect(f"tcp://192.168.0.132:{port}")  # Connect to server

    metadata = socket.recv_pyobj()
    payload = socket.recv()
    # print(metadata)
    # print(payload)
    np_dtype = dtype_mapping(metadata['dtype'])
    tensor = np.frombuffer(payload, dtype=np_dtype).reshape(metadata['shape'])
    tensor = torch.from_numpy(tensor)  # Convert NumPy array back to PyTorch tensor

    socket.close()
    context.term()

    return tensor

tensor = receive_tensor()
print(tensor)
