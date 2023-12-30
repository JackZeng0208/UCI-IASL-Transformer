import time

def model_evaluation(model,tokenizer,generate_token_num = 100):
    inputs = tokenizer('once upon a time', return_tensors="pt")["input_ids"].cuda()
    start = time.time()
    outputs = model.generate(inputs, max_new_tokens=generate_token_num)
    end = time.time()
    elapse_time = end - start
    print(f'output: {tokenizer.decode(outputs[0])}')
    print(f'token per second for generate {generate_token_num} tokens = {generate_token_num/elapse_time}\n ')