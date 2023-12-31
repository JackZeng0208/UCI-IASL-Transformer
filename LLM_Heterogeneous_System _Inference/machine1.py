# Imports
# Note: This is orin's code
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Initialize the distributed environment
def init_process():
    os.environ['MASTER_ADDR'] = '128.195.55.253'  # IP of Machine 1
    os.environ['MASTER_PORT'] = '8233'        # A chosen port
    dist.init_process_group(backend='gloo', rank=0, world_size=2) # Since it's an orin rather than GPU from PCIE

# Load and partition the model
def load_and_partition_model():
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", device_map="cuda", trust_remote_code=True)
    num_layers = len(model.transformer.h)
    model.transformer.h = model.transformer.h[:num_layers // 2]
    return model.to('cuda:0')

# Prepare input data
def prepare_input():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt").input_ids
    return inputs.to('cuda:0')

# Main function
def main():
    init_process()
    model = load_and_partition_model()
    input_ids = prepare_input()

    with torch.no_grad():
        outputs = model(input_ids)

    # Send outputs to Machine 2
    dist.send(tensor=outputs, dst=1)

if __name__ == "__main__":
    main()
