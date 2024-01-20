import os
import sys
import time
import torch
from pathlib import Path
from typing import List, Optional

# include root project path in sys
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from orin.config import OPTConfig
from orin.generate import generate, speculative_decoding
from orin.model import build_opt_model
from utils.evaluators import InferenceEvaluator


def main(
    # tasks: List[str],      evaluate tasks where user could specify token generation time, memory usage
    prompt: str,
    speculate_k: int,
    ip: str,
    port: int,
    top_k: float,
    top_p: int,
    compile: bool = False,
    model_size: str = "125M",
    max_seq_length: Optional[int] = 50,
):
    model, tokenizer, config = build_opt_model(model_size=model_size)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_tokens = input_ids
    rounds, left = int(max_seq_length / speculate_k), max_seq_length % speculate_k
    for _ in range(rounds):
        output_tokens = speculative_decoding(
            draft_model=model,
            input_ids=output_tokens,
            speculate_k=speculate_k,
            ip=ip,
            port=port,
            top_k=top_k,
            top_p=top_p
        )

    output_tokens = speculative_decoding(
        draft_model=model, 
        input_ids=output_tokens, 
        speculate_k=left,
        ip=ip,
        port=port,
        top_k=top_k,
        top_p=top_p
    )
    
    print(output_tokens)
    print("------------------------------------------")
    print("Outputs: ", tokenizer.decode(output_tokens))
    
    # generated_text = generate(
    #     model,
    #     tokenizer,
    #     prompt=prompt,
    #     max_new_tokens=max_seq_length
    # )
    # print(f"prompt: {prompt}")
    # print("model output: ", generated_text)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    # TODO: add arguments for programmatically executing file
    parser.add_argument("--model_size", type=str, default="125M")
    parser.add_argument("--prompt", type=str, help="prompt to the model.")
    parser.add_argument("--compile", action="store_true", help="Whether to compile the model.")
    parser.add_argument("--speculate_k", type=int, default=5, help="number of generated tokens per speculate step.")
    parser.add_argument("--max_seq_length", type=int, default=50, help="maximum length sequence to evaluate.")
    parser.add_argument("--top_k", type=int, default=50, help="number of most likely picked tokens for top k sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="accumulative probability mass for top p sampling.")
    parser.add_argument("--ip", type=str, help="ip address of server.")
    parser.add_argument("--port", type=int, help="port number for connecting to server.")
    args = parser.parse_args()

    main(
        prompt=args.prompt,
        compile=args.compile,
        model_size=args.model_size,
        speculate_k=args.speculate_k,
        max_seq_length=args.max_seq_length,
        ip=args.ip,
        port=args.port,
        top_k=args.top_k,
        top_p=args.top_p
    )
