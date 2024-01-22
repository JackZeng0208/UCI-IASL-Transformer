# Imports
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Initialize the distributed environment
def init_process():
    os.environ['MASTER_ADDR'] = '128.195.55.253'  # IP of Machine 1
    os.environ['MASTER_PORT'] = '8233'        # A chosen port
    dist.init_process_group(backend='nccl', rank=1, world_size=2)

def load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    second_half = torch.nn.Sequential(*list(model.children())[len(list(model.children())) // 2:])
    return second_half, tokenizer

def decode_output(tokenizer, outputs):
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    init_process()
    model_name = "microsoft/phi-2"  # Replace with your model
    model, tokenizer = load_model_and_tokenizer(model_name)

    # The tensor size here should match the output size of the first half of the model on Machine 1
    received_tensor = torch.empty(1, 512, dtype=torch.long).cuda()  # Adjust the size accordingly

    # Receive output from Machine 1
    dist.recv(tensor=received_tensor, src=0)

    # Run the second half of the model
    outputs = model(received_tensor)
    
    # Decode the output to text
    generated_text = decode_output(tokenizer, outputs)
    print(generated_text)

if __name__ == "__main__":
    main()