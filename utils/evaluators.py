# code of evluators for both inference and speculative decoding

import GPUtil
import time
from typing import Optional, List, Tuple
import torch
from orin.config import OPTConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

import os
# avoid running tokenizer in parallel to cause forking current process, potential deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class InferenceEvaluator():
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: Optional[int] = None
    ):
        self._model = model
        self._tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def eval(
        self,
        task: str,
        prompt: str,
        compile: bool = False,
    ) -> Tuple[str, float, float, float]:
        assert (task == "time" or task == "memory")
        model = self._model
        model.to(OPTConfig.device)
        input_ids = self._tokenizer.encode(prompt, return_tensors='pt')
        outputs = input_ids.to(OPTConfig.device)
        # allow compile model for optimization
        if compile:
            model = torch.compile(
                model, mode="reduce-overhead", fullgraph=True)

        # TODO: add memory checking
        GPUs = GPUtil.getGPUs()
        inference_history = {}
        gpu_usage = []
        model.eval()
        with torch.no_grad():
            token_count = 0
            print(f"Prompt: {prompt}")
            start_time = time.time()
            outputs = model.generate(
                outputs, max_new_tokens=1, do_sample=False)
            time_to_first_token = time.time() - start_time
            token_count += 1

            total_time = 0
            for _ in range(2, self.max_seq_length + 1):
                start_time = time.time()
                outputs = model.generate(
                    outputs, max_new_tokens=1, do_sample=False)
                token_count += 1
                next_token = outputs[0][-1]
                time_taken = time.time() - start_time
                total_time += time_taken

                # if generated token is end of sentence token, stop the inference
                if next_token.item() == self._tokenizer.eos_token_id:
                    break

            latency = time_to_first_token + total_time
            time_per_output_token = total_time / outputs.shape[1]
            # decode the generated token ids to text
            generated_text = self._tokenizer.decode(
                outputs[0][input_ids.shape[1] + 1:])
            inference_history["TTFT"] = time_to_first_token
            inference_history["TPOT"] = time_per_output_token
            inference_history["Latency"] = latency
        return generated_text, inference_history
