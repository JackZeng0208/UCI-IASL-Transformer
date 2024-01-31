# code for speculative decoder that allow edge devices to send requests to server for speculative decoding

import torch
import requests
from pathlib import Path
from typing import List, Optional
from utils.evaluators import InferenceEvaluator


class SpeculativeDecoder:
    def __init__(self, server_ip: str, port: int):
        self.server_url = f"{server_ip}:{port}/decode"

    def get_target_output(self, top_k: int, top_p: float, input_ids: torch.Tensor, draft_tokens: List[torch.Tensor]):
        response = requests.post(
            self.server_url,
            json={
                "draft_tokens": [token.item() for token in draft_tokens],
                "input_tokens": [id.item() for id in input_ids[0]],
                "top_k": top_k,
                "top_p": top_p
            }
        )
        print(response)
        if response.status_code == 200:
            return response.json().get("target_outputs")
        else:
            response.raise_for_status()
            return []

    def eval():
        pass
