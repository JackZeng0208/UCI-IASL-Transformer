# Imports
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Initialize the distributed environment
def init_process():
    os.environ['MASTER_ADDR'] = '66.42.104.193'  # IP of Machine 1
    os.environ['MASTER_PORT'] = '8233'        # A chosen port
    dist.init_process_group(backend='nccl', rank=0, world_size=2)

def load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    first_half = torch.nn.Sequential(*list(model.children())[:len(list(model.children())) // 2])
    return first_half, tokenizer

def process_input(tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')
    return inputs['input_ids'].cuda()

def main():
    init_process()
    model_name = "gpt2"  # Replace with your model
    model, tokenizer = load_model_and_tokenizer(model_name)

    input_text = "introduce yourself"
    input_ids = process_input(tokenizer, input_text)

    outputs = model(input_ids)
    print(outputs.shape)
    # Send output to Machine 2
    dist.send(tensor=outputs, dst=1)

if __name__ == "__main__":
    main()