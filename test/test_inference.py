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
    # tasks: List[str],
    prompt: str,
    compile: bool = False,
    model_size: str = "125M",
    max_seq_length: Optional[int] = 100
):
    model, tokenizer, config = build_opt_model(model_size=model_size)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_tokens = speculative_decoding(model, input_ids, speculative_k=5)

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
    parser.add_argument('--tasks', nargs='+', type=str, default=[
                        "hellaswag"], help='list of lm-eluther tasks to evaluate usage: --tasks task1 task2')
    parser.add_argument('--compile', action='store_true',
                        help='Whether to compile the model.')
    parser.add_argument('--max_seq_length', type=int,
                        default=50, help='maximum length sequence to evaluate')
    args = parser.parse_args()

    main(
        prompt=args.prompt,
        compile=args.compile,
        model_size=args.model_size,
        max_seq_length=args.max_seq_length
    )
