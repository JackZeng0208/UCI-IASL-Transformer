import torch
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM
from utils import model_evaluation

model_name = "bigscience/bloom-560m"
# You could also use "meta-llama/Llama-2-70b-chat-hf" or any other supported model from 🤗 Model Hub

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, add_bos_token=False)
model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
model = model.cuda()
# inputs = tokenizer('A cat in French is "', return_tensors="pt")["input_ids"].cuda()
# outputs = model.generate(inputs, max_new_tokens=3)
# print(tokenizer.decode(outputs[0]))


    
model_evaluation(model=model,tokenizer=tokenizer,generate_token_num = 100)
model_evaluation(model=model,tokenizer=tokenizer,generate_token_num = 50)
model_evaluation(model=model,tokenizer=tokenizer,generate_token_num = 10)
# about 2.12 token/s on StableBeluga2