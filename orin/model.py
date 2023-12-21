# code for model architecture with kv cache module for inference optimization

import torch
import torch.nn as nn
import accelerate
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from config import OPTConfig
from typing import Tuple, Optional


class KVCache(nn.Module):
    def __init__(self, max_batch_size: int, max_seq_length: int, n_heads: int, head_dim: int, dtype = OPTConfig.dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out

class EfficientModel(PreTrainedModel):
    def __init__(self, config: PretrainedConfig, *args, **kwargs):
        super().__init__()
        self.config = config
    
    def setup_caches(self, max_seq_length: int):
        if self.config.max_length >= max_seq_length:
            return
        

# FIXME: cannot use flash attention for now, only support CUDA version>11.6 (we are on 11.4)
def build_opt_model(model_size: str, model_name: str = "facebook/opt-") -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_size = model_size.lower()
    assert(model_size == "125m" or model_size == "350m" or model_size == "1.3b" 
           or model_size == "2.7b" or model_size == "6.7b")
    config = AutoConfig.from_pretrained(model_name + model_size)
    model = AutoModelForCausalLM.from_pretrained(model_name + model_size, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name + model_size)
    return model, tokenizer

# Note: only include draft model here since we will send request to server to handle problematic decoding tokens
def speculative_decode(
    draf_model: PreTrainedModel, 
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    device = OPTConfig.device
    pass

torch.no_grad()
def generate(
    model: PreTrainedModel,
    prompt: torch.Tensor,
    max_new_tokens: int,
):
    pass