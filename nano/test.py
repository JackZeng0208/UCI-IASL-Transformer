import torch 
print(torch.tensor(1))
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# # Load pre-trained model (weights)
# model = GPT2LMHeadModel.from_pretrained("gpt2")

# # Encode text input
# input_ids = tokenizer.encode("Hello, my name is", return_tensors="pt")

# # Generate text
# output = model.generate(input_ids, max_length=50)

# # Decode and print the output text
# print(tokenizer.decode(output[0], skip_special_tokens=True))
