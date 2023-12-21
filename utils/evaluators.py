# code of evluators for both inference and speculative decoding

import sys
import time
from typing import Optional, List


class InferenceEvaluator():
    def __init__(
        self, 
        model,
        tokenizer,
        max_seq_length: Optional[int] = None
    ):
        self._model = model
        self._tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def eval(
        tasks: List[str],
        prompts: List[str]
    ):
        pass
    
    
class SpeculativeEvaluator():
    def __init__(self):
        pass

    def eval(self, draft_text: str):
        pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')
    
    # TODO: add arguments for programmatically executing file
    parser.add_argument("", )
    args = parser.parse_args()
    
    
    