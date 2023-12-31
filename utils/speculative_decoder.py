# code for speculative decoder that allow edge devices to send requests to server for speculative decoding

from pathlib import Path
from typing import List, Optional
import requests

from evaluators import InferenceEvaluator


class SpeculativeDecoder:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.evaluator = InferenceEvaluator()

    def refine_text(self, draft_text):
        response = requests.post(self.server_url, json={
                                 "draft_text": draft_text})
        if response.status_code == 200:
            return response.json().get("refined_text", "")
        else:
            response.raise_for_status()

    def eval():
        pass
